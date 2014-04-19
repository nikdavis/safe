#!/bin/bash

SECONDS_TO_CAP=60

function timestamp() {
	date +"%Y%m%d_%H%M%S"
}

for ((;;)) do
	echo STARTING TO CAPTURE ${SECONDS_TO_CAP} SECOND\(S\)
	./main capture${SECONDS_TO_CAP}_`eval timestamp`.avi ${SECONDS_TO_CAP}
done 
