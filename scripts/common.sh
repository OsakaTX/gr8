# sourced by the other scripts: loads config/cluster.env and defines helpers
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CFG=${G6_CLUSTER:-$REPO/config/cluster.env}
[ -f "$CFG" ] || { echo "missing $CFG (copy config/cluster.env.example)" >&2; exit 2; }
# shellcheck disable=SC1090
. "$CFG"
read -r -a HOST <<< "$HOSTS"
NRANK=${#HOST[@]}
S="ssh -o BatchMode=yes"
on() { local r=$1; shift; $S "${HOST[$r]}" "$@"; }   # on <rank> <cmd>: run a command on that rank's node
