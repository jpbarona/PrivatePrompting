import os

from tests.e2e.conftest import E2EServices
from tests.e2e.orchestrate import require_exit_zero, start_process


def test_e2e_inference(e2e_services: E2EServices) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    parent_cmd = [
        e2e_services.python,
        "inference/parent_client.py",
        "--model-name",
        e2e_services.model_name,
        "--prompt",
        e2e_services.prompt,
        "--num-new-tokens",
        str(e2e_services.num_new_tokens),
        "--k",
        str(e2e_services.k),
        "--num-middle-partitions",
        str(e2e_services.num_middle_partitions),
        "--host-ip",
        e2e_services.host_ip,
        "--dht-port",
        str(e2e_services.dht_port_parent),
        "--bootstrap-maddr",
        e2e_services.bootstrap_maddr,
        "--w1-dht-key",
        e2e_services.dht_key_w1,
        "--w2-dht-key",
        e2e_services.dht_key_w2,
        "--max-seq",
        str(e2e_services.max_seq),
    ]
    parent = start_process(
        "parent",
        parent_cmd,
        cwd=str(e2e_services.repo_root),
        env=env,
    )
    parent.process.wait(timeout=e2e_services.parent_timeout_s)
    parent.thread.join(timeout=2.0)
    require_exit_zero(parent)

    assert any(line.strip() == "match: True" for line in parent.lines), "parent did not report 'match: True'"
    assert e2e_services.w1_handle.process.poll() is None, "w1 exited unexpectedly"
    assert e2e_services.w2_handle.process.poll() is None, "w2 exited unexpectedly"
