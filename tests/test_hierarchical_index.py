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
