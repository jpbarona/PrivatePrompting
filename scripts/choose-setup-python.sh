#!/bin/sh
set -e
resolve() {
	p="$1"
	while [ -L "$p" ]; do
		d=$(dirname "$p")
		p=$(readlink "$p")
		case "$p" in
		/*) ;;
		*) p="$d/$p";;
		esac
	done
	echo "$p"
}

if [ -n "$1" ] && [ "$(echo "$1" | cut -c1)" = "/" ]; then
	echo "$1"
	exit 0
fi

p=$(command -v python3 2>/dev/null) || true
if [ -z "$p" ]; then
	echo "python3"
	exit 0
fi

r=$(resolve "$p")
if echo "$r" | grep -q '.local/share/uv'; then
	for alt in /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
		[ -x "$alt" ] || continue
		ralt=$(resolve "$alt")
		echo "$ralt" | grep -q '.local/share/uv' && continue
		echo "$alt"
		exit 0
	done
	echo "python3"
else
	echo "$p"
fi
