for psr in $(cat ../../sims_epta+inpta_radiometer-wn/25PSR_list.txt); do
	echo { > ${psr}_noise2.json
	grep ${psr} crn_noise2.json >> ${psr}_noise2.json
	echo } >> ${psr}_noise2.json
done	
