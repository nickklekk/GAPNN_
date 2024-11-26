#!/bin/bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/data/zhaoxianjia/lib/htslib:/data/jiangheling/00_Biosoft/lib"

if [ $# -ne 3 ]
then
        echo "Usage: bash `basename $0` <prefix> <ref.fa> <query.fa>"
exit
fi
prefix=$1
ref=$(realpath $2)
query=$(realpath $3)

ln -sf $ref .
ln -sf $query .
/data/zhaoxianjia/software/mummer/nucmer -t 52 --prefix ${prefix} ${ref} ${query}

/data/zhaoxianjia/software/mummer/delta-filter -i 90 -l 10000 -q ${prefix}.delta > ${prefix}_i90_l10k.delta.filter
/data/zhaoxianjia/software/mummer/mummerplot --png -p ${prefix} ${prefix}_i90_l10k.delta.filter -R ${ref} -Q ${query} --filter

/data/zhaoxianjia/software/mummer/show-coords -r -T ${prefix}_i90_l10k.delta.filter >${prefix}_i90_l10k.coords
