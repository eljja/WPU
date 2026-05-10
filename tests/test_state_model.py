from wpu.core.state import Branch, DeltaState, Event, Relation, WorldObject, WorldState
from wpu.memory.state_store import StateStore


def test_world_state_json_roundtrip_and_delta_overlay() -> None:
    state = WorldState()
    state.add_object(WorldObject("cup_001", "cup", {"position": [0.0, 0.0, 1.0]}, 0.9))
    state.add_relation(Relation("cup_001", "table_001", "on_top_of", 0.8, 0.7))

    payload = state.to_json()
    loaded = WorldState.from_json(payload)
    assert loaded.objects["cup_001"].type == "cup"

    delta = DeltaState(time=1.0)
    delta.record_object("cup_001", {"position": [0.1, 0.0, 1.0]})
    branch = Branch("branch_A", "base", 0.6, delta)
    overlaid = loaded.overlay_branch(branch)
    assert overlaid.objects["cup_001"].attributes["position"] == [0.1, 0.0, 1.0]


def test_state_store_apply_event_creates_delta_without_mutating_base() -> None:
    state = WorldState(objects={"cup_001": WorldObject("cup_001", "cup", {"position": [0, 0, 0]}, 0.9)})
    store = StateStore(state)
    event = Event("object_moved", "cup_001", {"position": [1, 0, 0]}, confidence=0.5, time=1.0)

    delta = store.apply_event(event)

    assert delta.object_updates["cup_001"]["position"] == [1, 0, 0]
    assert state.objects["cup_001"].attributes["position"] == [0, 0, 0]
    assert delta.object_updates["cup_001"]["confidence"] == 0.45
