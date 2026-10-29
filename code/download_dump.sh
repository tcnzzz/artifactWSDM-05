#!/bin/bash
# Parallel ranged download of the HotpotQA Wikipedia intro dump (official Stanford host).
URL=https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
OUT=${1:-../data/raw}; N=${2:-1}; SIZE=1553565403   # use N=1 (one connection); the original run used N=32, see DATA.md
mkdir -p "$OUT/parts"; CH=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  s=$((i*CH)); e=$((s+CH-1)); [ $e -ge $SIZE ] && e=$((SIZE-1))
  f=$(printf "%s/parts/part_%03d" "$OUT" $i)
  ( have=0; [ -f "$f" ] && have=$(stat -f %z "$f")
    need=$((e-s+1))
    while [ $have -lt $need ]; do
      curl -s -A "research-crawl-replication (contact via paper)" --retry 5 -r $((s+have))-$e "$URL" >> "$f"; have=$(stat -f %z "$f"); done ) &
done
wait
cat "$OUT"/parts/part_* > "$OUT/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
shasum -a 256 "$OUT"/*.tar.bz2 > "$OUT/SHA256SUMS"; rm -r "$OUT/parts"
