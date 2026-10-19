#!/bin/bash
# Run the COMP 536 project's own tests, not vLLM's upstream suite.
#
#   tools/run_project_tests.sh                 # every milestone
#   tools/run_project_tests.sh m2              # Milestone 2 only (m1, m2, m3)
#   tools/run_project_tests.sh m1 -k prefix    # extra arguments go to pytest
#   tools/run_project_tests.sh check-markers   # fail if a test has no m1/m2/m3
#
# Each test module declares its milestone with `pytestmark = pytest.mark.m2`
# (markers are registered in pyproject.toml). project-ci runs this script
# once per milestone (specs/001-ci-cd).
set -euo pipefail

cd "$(dirname "$0")/.."
# The platform is chosen when vllm is imported, so this must be set first.
export VLLM_USE_SIMULATOR=1 VLLM_NO_USAGE_STATS=1

# Folders that hold only project tests. Add new ones here.
PROJECT_TEST_DIRS="tests/simulator tests/client_simulator tests/analysis tests/eviction"

dirs=()
for d in $PROJECT_TEST_DIRS; do
    if [ -d "$d" ]; then
        dirs+=("$d")
    fi
done

mode=${1:-all}
if [ $# -gt 0 ]; then
    shift
fi

pytest_cmd=(python -m pytest "${dirs[@]}"
            -o asyncio_default_fixture_loop_scope=function)

case "$mode" in
    all)
        selection=()
        ;;
    m1 | m2 | m3)
        selection=(-m "$mode")
        ;;
    check-markers)
        set +e
        output=$("${pytest_cmd[@]}" -m "not m1 and not m2 and not m3" \
                 --collect-only -q 2>&1)
        status=$?
        set -e
        if [ $status -eq 5 ]; then
            echo "Every project test is marked m1, m2 or m3."
            exit 0
        fi
        echo "$output"
        if [ $status -eq 0 ]; then
            echo "The tests above have no milestone marker. Add" \
                 "'pytestmark = pytest.mark.m<N>' to their module."
            exit 1
        fi
        exit $status
        ;;
    *)
        echo "usage: $0 [all|m1|m2|m3|check-markers] [pytest args...]" >&2
        exit 2
        ;;
esac

set +e
# ${selection[@]+...} keeps `set -u` happy on bash 3.2 (macOS) when empty.
"${pytest_cmd[@]}" ${selection[@]+"${selection[@]}"} "$@"
status=$?
set -e
if [ $status -eq 5 ]; then
    # pytest exits 5 when nothing was collected: no tests for this milestone.
    echo "No $mode tests yet."
    exit 0
fi
exit $status
