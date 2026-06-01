
from campus.core import Campus

import pytest


@pytest.mark.api
def test_train_new_expert(g_data):
    campus = Campus(
        global_task="Write poems.",
        save_path="./tests/data",
        model=g_data["model"]
    )

    campus.train_new_expert("Nature poems", "Poems about the nature.")
    assert True
    
    
def test_list_all_experts():
    pass