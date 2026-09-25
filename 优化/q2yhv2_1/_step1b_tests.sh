PY=/data2/hy/anaconda3/envs/secgpt-vllm/bin/python
cd "/data2/hy/smgs/问题二/q2_ds_optimization_v2_1_20260925"
$PY tests/test_protocol.py -v > /tmp/q2v21_tests.log 2>&1
echo "EXIT=$?"
tail -n 25 /tmp/q2v21_tests.log