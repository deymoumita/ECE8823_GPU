#!/bin/bash

infile=$1

HARP_ARCH=4w8/1/4/1

HARPTOOL=../../harptool2/src/harptool

filename=$(basename "$infile" .s)
infile_HOF=${filename}.HOF
infile_BIN=${filename}.bin

echo "$HARPTOOL -A -a $HARP_ARCH -o runtime.HOF runtime.s"
$HARPTOOL -A -a $HARP_ARCH -o runtime.HOF runtime.s || exit 1

echo "$HARPTOOL -A -a $HARP_ARCH -o $infile_HOF $infile"
$HARPTOOL -A -a $HARP_ARCH -o $infile_HOF $infile || exit 1

echo "$HARPTOOL -L -a $HARP_ARCH -o $infile_BIN $infile_HOF runtime.HOF"
$HARPTOOL -L -a $HARP_ARCH -o $infile_BIN $infile_HOF runtime.HOF || exit 1
