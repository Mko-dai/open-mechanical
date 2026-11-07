from openmech.history import History
from openmech.project import Project
from openmech.geometry import new_feature
from openmech.solver import BC


def test_history_undo_redo_branch():
    p = Project()
    h = History()
    h.reset(p.to_dict())
    p.geometry["features"].append(new_feature("box"))
    assert h.record(p.to_dict())
    assert h.items[-1]["label"] == "作成: 直方体"
    p.bcs.append(BC("fixed", [0]))
    h.record(p.to_dict())
    assert h.items[-1]["label"] == "追加: 固定支持"
    assert not h.record(p.to_dict())          # 変化なし → 記録しない
    # 1つ目（開始）まで戻る
    h.pos = 0
    q = Project.from_dict(h.state(h.pos))
    assert q.geometry["features"] == [] and not q.bcs
    assert h.can_redo()
    # 戻った位置から別の操作 → 先の履歴は捨てる
    q.geometry["features"].append(new_feature("cylinder"))
    h.record(q.to_dict())
    assert len(h.items) == 2 and h.items[-1]["label"] == "作成: 円柱" and not h.can_redo()
