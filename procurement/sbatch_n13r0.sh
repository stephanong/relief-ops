#!/bin/sh

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G


DIR=data
OUTPUTDIR=output
CONF=n13r0 # nothing, or n13r0, or whatever configuration created
INST=100s0 # eg. 100s0, or 1000s0, or whatever data set created

CONFIG=simconfig${CONF}


MODEL=realhist5 
TEST=realhist5
./procure_simulator.py -c ${OUTPUTDIR}/${CONFIG}.json -i ${DIR}/${MODEL}.csv -o ${OUTPUTDIR}/output${CONF}-${MODEL}-${TEST}-basic.csv       -r basic
./procure_simulator.py -c ${OUTPUTDIR}/${CONFIG}.json -i ${DIR}/${MODEL}.csv -o ${OUTPUTDIR}/output${CONF}-${MODEL}-${TEST}-affine.csv      -r affine
./procure_simulator.py -c ${OUTPUTDIR}/${CONFIG}.json -i ${DIR}/${MODEL}.csv -o ${OUTPUTDIR}/output${CONF}-${MODEL}-${TEST}-pct100.csv      -r percentile -pr 100
./procure_simulator.py -c ${OUTPUTDIR}/${CONFIG}.json -i ${DIR}/${MODEL}.csv -o ${OUTPUTDIR}/output${CONF}-${MODEL}-${TEST}-stoch.csv       -r stoch

./procure_simulator.py -c ${OUTPUTDIR}/${CONFIG}.json -i ${DIR}/${TEST}.csv ${DIR}/${TEST}.csv -o ${OUTPUTDIR}/output${CONF}-${TEST}-clairvoyant.csv -r clairvoyant


MODEL=realhist5
TEST=gentest${INST}
./procure_simulator.py -c ${OUTPUTDIR}/${CONFIG}.json -i ${DIR}/${MODEL}.csv ${DIR}/${TEST}.csv -o ${OUTPUTDIR}/output${CONF}-${MODEL}-${TEST}-basic.csv       -r basic
./procure_simulator.py -c ${OUTPUTDIR}/${CONFIG}.json -i ${DIR}/${MODEL}.csv ${DIR}/${TEST}.csv -o ${OUTPUTDIR}/output${CONF}-${MODEL}-${TEST}-affine.csv      -r affine
./procure_simulator.py -c ${OUTPUTDIR}/${CONFIG}.json -i ${DIR}/${MODEL}.csv ${DIR}/${TEST}.csv -o ${OUTPUTDIR}/output${CONF}-${MODEL}-${TEST}-pct100.csv      -r percentile -pr 100
./procure_simulator.py -c ${OUTPUTDIR}/${CONFIG}.json -i ${DIR}/${MODEL}.csv ${DIR}/${TEST}.csv -o ${OUTPUTDIR}/output${CONF}-${MODEL}-${TEST}-stoch.csv       -r stoch

./procure_simulator.py -c ${OUTPUTDIR}/${CONFIG}.json -i ${DIR}/${TEST}.csv ${DIR}/${TEST}.csv -o ${OUTPUTDIR}/output${CONF}-${TEST}-clairvoyant.csv -r clairvoyant


MODEL=genhist${INST}
TEST=gentest${INST}
./procure_simulator.py -c ${OUTPUTDIR}/${CONFIG}.json -i ${DIR}/${MODEL}.csv ${DIR}/${TEST}.csv -o ${OUTPUTDIR}/output${CONF}-${MODEL}-${TEST}-basic.csv       -r basic
./procure_simulator.py -c ${OUTPUTDIR}/${CONFIG}.json -i ${DIR}/${MODEL}.csv ${DIR}/${TEST}.csv -o ${OUTPUTDIR}/output${CONF}-${MODEL}-${TEST}-affine.csv      -r affine
./procure_simulator.py -c ${OUTPUTDIR}/${CONFIG}.json -i ${DIR}/${MODEL}.csv ${DIR}/${TEST}.csv -o ${OUTPUTDIR}/output${CONF}-${MODEL}-${TEST}-pct90.csv       -r percentile -pr 90
./procure_simulator.py -c ${OUTPUTDIR}/${CONFIG}.json -i ${DIR}/${MODEL}.csv ${DIR}/${TEST}.csv -o ${OUTPUTDIR}/output${CONF}-${MODEL}-${TEST}-pct100.csv      -r percentile -pr 100
./procure_simulator.py -c ${OUTPUTDIR}/${CONFIG}.json -i ${DIR}/${MODEL}.csv ${DIR}/${TEST}.csv -o ${OUTPUTDIR}/output${CONF}-${MODEL}-${TEST}-stoch.csv       -r stoch
./procure_simulator.py -c ${OUTPUTDIR}/${CONFIG}.json -i ${DIR}/${MODEL}.csv ${DIR}/${TEST}.csv -o ${OUTPUTDIR}/output${CONF}-${MODEL}-${TEST}-stochcvarw90.csv -r stochcvarw -pa 0.90
./procure_simulator.py -c ${OUTPUTDIR}/${CONFIG}.json -i ${DIR}/${MODEL}.csv ${DIR}/${TEST}.csv -o ${OUTPUTDIR}/output${CONF}-${MODEL}-${TEST}-stochcvarw95.csv -r stochcvarw -pa 0.95
./procure_simulator.py -c ${OUTPUTDIR}/${CONFIG}.json -i ${DIR}/${MODEL}.csv ${DIR}/${TEST}.csv -o ${OUTPUTDIR}/output${CONF}-${MODEL}-${TEST}-stochcvar90.csv  -r stochcvar  -pa 0.90
./procure_simulator.py -c ${OUTPUTDIR}/${CONFIG}.json -i ${DIR}/${MODEL}.csv ${DIR}/${TEST}.csv -o ${OUTPUTDIR}/output${CONF}-${MODEL}-${TEST}-stochcvar95.csv  -r stochcvar  -pa 0.95

#./procure_simulator.py -c ${OUTPUTDIR}/${CONFIG}.json -i ${DIR}/${TEST}.csv ${DIR}/${TEST}.csv -o ${OUTPUTDIR}/output${CONF}-${TEST}-clairvoyant.csv -r clairvoyant ## no need, as already done above

