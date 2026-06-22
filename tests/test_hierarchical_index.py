from wpu.core.hierarchy import HierarchicalWorldState, WorldCausalIndex, WorldCausalQuery
from wpu.core.state import Event, Relation, WorldObject, WorldState


def test_hierarchical_world_state_assigns_region_membership() -> None:
    state = WorldState()
    state.add_object(WorldObject("cup", "cup", {"position": [0.0, 0.0, 0.0]}))
    state.add_object(WorldObject("table", "table", {"position": [0.1, 0.0, 0.0]}))
    hierarchy = HierarchicalWorldState(state)

    hierarchy.add_region("kitchen", parent_id="apartment")
    hierarchy.assign_object("cup", "kitchen")
    hierarchy.assign_object("table", "kitchen")

    assert hierarchy.region_objects("apartment", recursive=True) == ["cup", "table"]
    assert hierarchy.object_to_region["cup"] == "kitchen"


def test_world_causal_index_keeps_large_background_out_of_event_slice() -> None:
    state = WorldState()
    state.add_object(WorldObject("cup", "cup", {"position": [0.0, 0.0, 0.0]}, last_updated=0.5))
    state.add_object(WorldObject("table", "table", {"position": [0.1, 0.0, 0.0]}))
    state.add_object(WorldObject("hand", "hand", {"position": [0.2, 0.0, 0.0]}))
    state.add_relation(Relation("cup", "table", "on"))
    state.add_relation(Relation("hand", "cup", "near"))
    for index in range(100):
        state.add_object(WorldObject(f"bg_{index}", "background", {"position": [100.0 + index, 0.0, 0.0]}))
    event = Event("touch", "cup", {"force": 0.5}, time=1.0)

    causal_slice = WorldCausalIndex(state).query(
        WorldCausalQuery(event=event, max_objects=8, relation_depth=1, spatial_radius=0.5)
    )

    assert causal_slice.total_objects == 103
    assert causal_slice.affected_fraction < 0.08
    assert causal_slice.object_ids[:3] == ["cup", "table", "hand"]
    assert not any(object_id.startswith("bg_") for object_id in causal_slice.object_ids)
    assert causal_slice.reason_by_object["cup"] == {"event_target", "recent_change"}


def test_world_causal_index_reports_paths_and_scoped_retrieval_cost() -> None:
    state = WorldState()
    state.add_object(WorldObject("cup", "cup", {"position": [0.0, 0.0, 0.0]}, last_updated=0.5))
    state.add_object(WorldObject("table", "table", {"position": [0.1, 0.0, 0.0]}))
    state.add_object(WorldObject("hand", "hand", {"position": [0.2, 0.0, 0.0]}))
    state.add_relation(Relation("cup", "table", "on"))
    state.add_relation(Relation("hand", "cup", "near"))
    hierarchy = HierarchicalWorldState(state)
    hierarchy.add_region("active", parent_id="world")
    for object_id in ["cup", "table", "hand"]:
        hierarchy.assign_object(object_id, "active")
    for index in range(200):
        object_id = f"bg_{index}"
        state.add_object(WorldObject(object_id, "background", {"position": [float(index), 99.0, 0.0]}))
        hierarchy.add_region("background", parent_id="world")
        hierarchy.assign_object(object_id, "background")

    causal_slice = WorldCausalIndex(state, hierarchy).query(
        WorldCausalQuery(event=Event("touch", "cup", time=1.0), max_objects=8, spatial_radius=0.5)
    )

    assert causal_slice.relation_path_by_object["hand"] == ["cup", "hand"]
    assert causal_slice.retrieval_metrics["candidate_scope_size"] == 3
    assert causal_slice.retrieval_metrics["objects_examined"] < len(state.objects)
    assert causal_slice.retrieval_metrics["relations_examined"] == 2


def test_world_causal_index_rejects_low_confidence_spurious_relations() -> None:
    state = WorldState()
    state.add_object(WorldObject("cup", "cup", {"position": [0.0, 0.0, 0.0]}))
    state.add_object(WorldObject("table", "table", {"position": [0.1, 0.0, 0.0]}))
    state.add_object(WorldObject("distractor", "background", {"position": [9.0, 0.0, 0.0]}))
    state.add_relation(Relation("cup", "table", "on", confidence=0.95))
    state.add_relation(Relation("cup", "distractor", "spurious", confidence=0.2))

    causal_slice = WorldCausalIndex(state).query(
        WorldCausalQuery(
            event=Event("touch", "cup", time=1.0),
            include_recent=False,
            include_uncertain=False,
            spatial_radius=0.2,
            min_relation_confidence=0.5,
        )
    )

    assert "table" in causal_slice.object_ids
    assert "distractor" not in causal_slice.object_ids
    assert causal_slice.retrieval_metrics["relations_rejected_low_confidence"] == 1
    assert causal_slice.retrieval_metrics["escalation_required"] == 1


def test_world_causal_index_marks_escalation_when_true_relation_confidence_is_low() -> None:
    state = WorldState()
    state.add_object(WorldObject("cup", "cup", {"position": [0.0, 0.0, 0.0]}))
    state.add_object(WorldObject("table", "table", {"position": [0.1, 0.0, 0.0]}))
    state.add_relation(Relation("cup", "table", "on", confidence=0.2))
    hierarchy = HierarchicalWorldState(state)
    hierarchy.add_region("active")
    hierarchy.assign_object("cup", "active")
    hierarchy.assign_object("table", "active")

    causal_slice = WorldCausalIndex(state, hierarchy).query(
        WorldCausalQuery(
            event=Event("touch", "cup", time=1.0),
            include_recent=False,
            include_uncertain=False,
            spatial_radius=0.0,
            min_relation_confidence=0.5,
        )
    )

    assert "table" in causal_slice.object_ids
    assert causal_slice.reason_by_object["table"] == {"same_region"}
    assert causal_slice.retrieval_metrics["relations_rejected_low_confidence"] == 1
    assert causal_slice.retrieval_metrics["escalation_required"] == 1
