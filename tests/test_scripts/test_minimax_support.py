from pathlib import Path


def test_minimax_chat_template_registered():
    template_path = Path(__file__).parent.parent.parent.joinpath(
        "specforge", "data", "template.py"
    )
    content = template_path.read_text()

    assert 'name="minimax-m2"' in content
    assert 'assistant_header="]~b]ai\\n"' in content
    assert 'user_header="]~b]user\\n"' in content
    assert 'end_of_turn_token="[e~["' in content


def test_minimax_example_script_contains_required_flags():
    script_path = (
        Path(__file__)
        .parent.parent.parent.joinpath("examples", "run_minimax_m2.5_eagle3_online.sh")
    )
    script = script_path.read_text()

    assert "TARGET_MODEL_BACKEND=${TARGET_MODEL_BACKEND:-sglang}" in script
    assert "--target-model-backend \"$TARGET_MODEL_BACKEND\"" in script
    assert "SGLANG_EP_SIZE=${SGLANG_EP_SIZE:-$TP_SIZE}" in script
    assert "--sglang-ep-size \"$SGLANG_EP_SIZE\"" in script
    assert "--trust-remote-code" in script
    assert "--chat-template minimax-m2" in script
    assert "TARGET_MODEL_PATH=${TARGET_MODEL_PATH:-/home/scratch.fengyul_coreai/model_ckpt/MiniMax-M2.5}" in script
    assert "--draft-model-config \"$ROOT_DIR/configs/minimax-m2.5-eagle3.json\"" in script
