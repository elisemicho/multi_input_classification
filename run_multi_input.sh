#PBS -l nodes=ssaling11:ppn=1:gpus=1,mem=2G,walltime=10000:00:00 -N I.just.need.one
GPUS=`cat ${PBS_GPUFILE} | perl -pe 's/[^-]+-gpu//g' | perl -pe 's/\s+/ /g' | perl -pe 's/,$//g'`

CODE_FOLDER=~/projects/VarDial2018/to_export/multi_input_modular

export PATH="/home/michon/anaconda2/bin:$PATH"

source activate py35

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64/:/home/michon/cuda/lib64/:/DEV/cuda/lib64/:/home/klein/lib/cuda/lib64/:/home/klein/cuda/lib64/

preprocess(){
	echo "############# preprocess @ "`date`" GPUS=$GPUS HOST=$HOST PWD="`pwd`
	python data_preprocess_simple.py
	echo "############# preprocess: DONE @ "`date` 
}

train(){
	echo "############# train @ "`date`" GPUS=$GPUS HOST=$HOST PWD="`pwd`
	python main.py
    echo "############# train: DONE @ "`date` 
}

cd $CODE_FOLDER

MODEL=acoustic_only
CORPUS=vardial2018
TIME=`date +"%Y-%m-%d_%H-%M-%S"`
HOST=`hostname`

# preprocess &> log.preprocess.$CORPUS.$TIME.$HOST.$GPUS
train &> log.$MODEL.$CORPUS.$TIME.$HOST.$GPUS