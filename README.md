Execution sequence:

  0. Add admin user: curl -s -X PUT http://localhost:5984/_config/admins/admin -d '"Celmdpqosx1!"'

	1. run the network
	python NetworkTopology.py
	
	2. Creat the Database
	python db_creat.py
		- To delete the Database
		  python db_delete.py
		- to check the Database
		  http://localhost:5984/_utils/index.html
		  
	3. REST input to update Database
	python Controller_restApi.py
		- To insert the input
		curl http://127.0.0.1:5000/route_id/2 -X PUT    [for Route 2]
		curl http://127.0.0.1:5000/route_id/1 -X PUT    [for Route 1]

	4. Run the Controller [from 001 to 005 - follow the controller order]
		ryu-manager Ctrl_TIFS.py --observe-link --ofp-tcp-listen-port 6653

		- To check the switch rule table
		  ovs-ofctl -O openflow14 dump-flows s1
	
	In HOST ( h3: 10.0.0.3 ): 
	# sudo arp -s 10.0.0.7 00:00:00:00:04:04

	For the IEEE TIFS paper,
	# nping --tcp 10.0.0.7 -c 1 --rate 1000 -p 1-60000

	In HOST ( h4: 10.0.0.4 ): 
	# sudo arp -s 10.0.0.3 00:00:00:00:03:03


	5. To send data-packets from Host-1 (H1) to Host-2 (H2) 

	In terminal
	-> curl http://127.0.0.1:5000/route_id/1 -X PUT  [it installs necessary rules in switches ]
	
	In HOST ( h1: 10.0.0.1 ): 
	# sudo arp -s 10.0.0.2 00:00:00:00:02:02
	# iperf -c 10.0.0.2 -u -t 2000              [ It sends UDP packets to H2]

	In HOST ( h2: 10.0.0.2 ): 
	# sudo arp -s 10.0.0.2 00:00:00:00:02:02
	# iperf -s -u -i 2 -t 2000                   [ It listens the UDP packets coming from H1]



Special Note:
	- Ctrl_TIFS.py is the controller code of IEEE TIFS 2018 paper that has the attack mitigation technique implemented.
	- Ctrl_TIFS_2.py is the controller code without having the attack mitigation technique implemented.
	- All incoming packets information are stored in "all-packet-information.txt" file.
	- To change the file format to PCAP file: results_file = open("results-ruleInsert-standard.txt", "a")  ->  standard.pcap", "a")

	CPU utilization check:-
		htop --user root -p 23636 -d 3



