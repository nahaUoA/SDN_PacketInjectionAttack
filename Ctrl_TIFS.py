
from operator import attrgetter
from ryu.base import app_manager
from ryu.lib.pack_utils import msg_pack_into
from ryu import utils
from ryu.controller.handler import HANDSHAKE_DISPATCHER, CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_4, ether
from ryu.lib import hub
from ryu.topology import event
from ryu.lib.packet import packet, ethernet, arp, icmp, ipv4, ipv6, tcp, udp
from ryu.lib.packet import ether_types
from ryu.topology.switches import LLDPPacket
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from webob import Response
from ryu.base import app_manager
from ryu.controller import ofp_event, dpset
from ryu.lib import dpid as dpid_lib
from couchdb.client import Server, Document
from ryu.topology import event, switches
from ryu.topology.api import get_switch, get_link
from collections import OrderedDict
from threading import Thread
import socket
import copy
import os
import time
import random

#Create connection object for CouchDB connection
#server = Server('http://localhost:5984')
server = Server('http://admin:Celmdpqosx1!@localhost:5984')
dblogger = server['common_logger']


# input 001 (un-important). Hosts IP addresses.
Host_01 = '10.0.0.1'
Host_02 = '10.0.0.2'
Host_03 = '10.0.0.3'

# input 002 (Ryu controller ID). It is the unique ID of the controller
Controller_ID = 1

# input 003 ( Target switch ID and its port number). It depends on the network topology.
affected_switch = 3
affected_port = 1
drop_port = 4

# Network information of all hosts available to all controllers
ctrl_log_table =	{
  1: {"MAC": '00:00:00:00:01:01', "IP": '10.0.0.1', "PORT": 1001 },
  2: {"MAC": '00:00:00:00:02:02', "IP": '10.0.0.2', "PORT": 1001 }  }

# Controller will maintain a text file to keep the information of all incoming packets
results_file = open("all-packet-information.pcap", "a")
results_file.write("SW-ID  Prt-no  Pkt-no       Time            MAC-SrcAddr        EthType     MAC-DestAddr          IP-SrcAddr   IP-DstAddr    TCP_SrcPrt      TCP_DstPrt\n" )
results_file.close()


class SimpleSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_4.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SimpleSwitch13, self).__init__(*args, **kwargs)

        self.mac_to_port = {}
        self.datapaths = {}
        self.topology_api_app = self
        self.topo_raw_switches = []
        self.topo_raw_links = []

        self.Rule_insert_1 = True
        self.Rule_insert_2 = True
        self.Rule_insert_3 = True
        self.Rule_insert_4 = True
        self.Rule_insert_5 = True
        self.rule_priority = 100
        self.routing_path = 0
        self.all_pkts = 0
        self.packet_block = False
        self.total_rule_no = 0
        self.present_time = 0
        self.past_time = 0
        self.flow_time_present = 0
        self.flow_time_past = 0
        self.packet_num_past = 0 
        self.flow_rate = 1
        self.target_time = 1
        self.rule_delete_time = 60
        self.rule_space_max = 10000 
        self.ctrl_input_in = 0
        self.ctrl_input_out = 0
        
# initialization of the controller role 
        self.gen_id = 0
        self.role_string_list = ['nochange', 'equal', 'master', 'slave', 'unknown']
# Set time in sec to start the main rule insertion/deletion function execution
        self.monitor_thread = hub.spawn_after(2, self._monitor)

# ...................................Network Topology Info...........................................
    @set_ev_cls(event.EventSwitchEnter)
    def handler_switch_enter(self, ev):
        self.topo_raw_switches = copy.copy(get_switch(self, None))
        self.topo_raw_links = copy.copy(get_link(self, None))
        switches = [switch.dp.id for switch in self.topo_raw_switches]
        links_list = get_link(self.topology_api_app, None)
        links = [(link.src.dpid, link.dst.dpid, {'port': link.src.port_no}) for link in links_list]
        self.AllSwitchesID = switches
        self.Alllinks = links
        self.logger.debug('All switches: %s,\nAll links: %s,\n', self.AllSwitchesID, self.Alllinks)
# ....................................................................................................

# ......................................... switch_in_handler .......................................
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        dpid = datapath.id
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
# ....................................................................................................

# ............................................ add_flow ..............................................
    def add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst)
        datapath.send_msg(mod)
# ....................................................................................................

# .................................. add_flow-for blocking packets ...................................
    def add_flow_blk(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        idle_timeout = self.rule_delete_time
        hard_timeout=0
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, idle_timeout=idle_timeout, hard_timeout=hard_timeout, priority=priority, match=match, instructions=inst)
        datapath.send_msg(mod)
# ....................................................................................................

# ............................................ delete_flow ............................................
    def delete_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        flow_del = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, command=ofproto.OFPFC_DELETE_STRICT, buffer_id=ofproto.OFP_NO_BUFFER, out_port=ofproto.OFPP_ANY, out_group=ofproto.OFPG_ANY, instructions=inst)
        datapath.send_msg(flow_del)
# ....................................................................................................

# ......................................... Controller role req/res .....................................

    def send_role_request(self, datapath, role, gen_id):
        dpid = datapath.id
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.logger.debug('Controller-role change request to switch-ID: %s', dpid)
        self.logger.debug('Controller-role: %s,  Generation-ID: %s \n', self.role_string_list[role], gen_id)
        req = parser.OFPRoleRequest(datapath, role, gen_id)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPRoleReply, MAIN_DISPATCHER)
    def role_reply_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofproto = dp.ofproto
        role = msg.role

        # unknown role
        if role < 0 or role > 3:
            role = 4

        if role == ofproto.OFPCR_ROLE_NOCHANGE:
            role = 'NOCHANGE'
        elif role == ofproto.OFPCR_ROLE_EQUAL:
            role = 'EQUAL'
        elif role == ofproto.OFPCR_ROLE_MASTER:
            role = 'MASTER'
        elif role == ofproto.OFPCR_ROLE_SLAVE:
            role = 'SLAVE'
        else:
            role = 'unknown'

        self.logger.debug('OFPRoleReply received,\n '
                        'role: %s, generation_id: %d', role, msg.generation_id)
# ......................................................................................................

# .......................................... Flow status request/response ............................

    def send_aggregate_stats_request(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        cookie = cookie_mask = 0
        out_port = drop_port
        match = parser.OFPMatch(eth_type=0x0800, in_port= affected_port)
        req = parser.OFPAggregateStatsRequest(datapath,0,ofproto.OFPTT_ALL,drop_port,ofproto.OFPG_ANY, cookie, cookie_mask,match)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPAggregateStatsReply, MAIN_DISPATCHER)
    def aggregate_stats_reply_handler(self, ev):
        body = ev.msg.body
        datapath = ev.msg.datapath
        dpid = datapath.id
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.logger.debug('AggregateStats: packet_count=%d byte_count=%d '
                          'flow_count=%d',
                          body.packet_count, body.byte_count,
                          body.flow_count)
        if (body.flow_count != 0):                        # Important: start counting the flow
            self.flow_time_present = time.time()
            self.logger.debug('Total %s rules are available in Switch %s at time %s', body.flow_count, dpid, time.time())
# .......................................................................................................

# .......................................... Port status request/response ............................

    def send_port_request(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        body = ev.msg.body
        datapath = ev.msg.datapath
        dpid = datapath.id
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self.logger.debug('datapath         port     '
                         'rx-pkts  rx-bytes rx-error '
                         'tx-pkts  tx-bytes tx-error '
                         'duration-sec  duration-nsec')
        self.logger.debug('---------------- -------- '
                         '-------- -------- -------- '
                         '-------- -------- -------- '
                         '-------------  -------------')
        for stat in sorted(body, key=attrgetter('port_no')):
            self.logger.debug('%016x %8x %8d %8d %8d %8d %8d %8d %13d %13d',
                             ev.msg.datapath.id, stat.port_no,
                             stat.rx_packets, stat.rx_bytes, stat.rx_errors,
                             stat.tx_packets, stat.tx_bytes, stat.tx_errors,
                             stat.duration_sec, stat.duration_nsec)
# .......................................................................................................

# ............................................. packet_in_handler ......................................
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        dpid = datapath.id
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth_pkt = pkt.get_protocol(ethernet.ethernet)
        arp_pkt = pkt.get_protocol(arp.arp)
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        tcp_pkt = pkt.get_protocol(tcp.tcp)
        udp_pkt = pkt.get_protocol(udp.udp)

        if msg.reason == ofproto.OFPR_APPLY_ACTION:
            reason = 'APPLY ACTION'
        elif msg.reason == ofproto.OFPR_INVALID_TTL:
            reason = 'INVALID TTL'
        elif msg.reason == ofproto.OFPR_ACTION_SET:
            reason = 'ACTION SET'
        elif msg.reason == ofproto.OFPR_GROUP:
            reason = 'GROUP'
        elif msg.reason == ofproto.OFPR_PACKET_OUT:
            reason = 'PACKET OUT'
        else:
            reason = 'unknown'

        self.logger.debug('OFPPacketIn received: '
                      'buffer_id=%x total_len=%d reason=%s '
                      'table_id=%d cookie=%d match=%s data=%s',
                      msg.buffer_id, msg.total_len, reason,
                      msg.table_id, msg.cookie, msg.match,
                      utils.hex_array(msg.data))
        self.logger.debug('in Switch: %s \nPacket information: %s\n\n', dpid, pkt)
        self.logger.debug('in Switch: %s \nswitch in-port: %s \nReason: %s\n\n', dpid, msg.match['in_port'], reason)

        if ( dpid == affected_switch and msg.match['in_port']== affected_port ):
            self.all_pkts = self.all_pkts + 1
            self.logger.debug('Switch-ID: %s, in-port: %s total incoming pkts: %s at time: %s\n', dpid, msg.match['in_port'], self.all_pkts, time.time())
            if (arp_pkt != None or ip_pkt != None):
                self.logger.info('Packet information: %s', pkt)
                self.logger.debug('eth-Packet information: %s', eth_pkt)
                self.logger.debug('eth-src: %s, eth-type: %s, eth-dst: %s\n', eth_pkt.src, eth_pkt.ethertype, eth_pkt.dst)
                self.logger.debug('ARP-Packet information: %s', arp_pkt)
                self.logger.debug('IP-Packet information: %s\n', ip_pkt)

                results_file = open("all-packet-information.txt", "a")
                results_file.write("%s %7s %8s %21s %19s %6s %22s %14s %12s %10s %13s\n" % (dpid, msg.match['in_port'], self.all_pkts, time.time(), eth_pkt.src, eth_pkt.ethertype, eth_pkt.dst, ip_pkt.src, ip_pkt.dst, tcp_pkt.src_port, tcp_pkt.dst_port))
                results_file.close()

                if (tcp_pkt != None):
                    self.logger.debug('tcp-src-port: %s, tcp-dst-port: %s', tcp_pkt.src_port, tcp_pkt.dst_port)
                if (udp_pkt != None):                
                    self.logger.debug('udp-src-port: %s, udp-dst-port: %s', udp_pkt.src_port, udp_pkt.dst_port)
                if (arp_pkt != None):
                    self.logger.debug('MAC-src: %s, IP-src: %s, proto: %s, MAC-dst: %s, IP-dst: %s', arp_pkt.src_mac, arp_pkt.src_ip, arp_pkt.proto, arp_pkt.dst_mac, arp_pkt.dst_ip)
                if (ip_pkt != None):
                    self.logger.debug('IP-src: %s, IP-dst: %s, operation codes: %s', ip_pkt.src, ip_pkt.dst, ip_pkt.proto)
                    i = 1
                    while ( i<= len(ctrl_log_table) ):
                        if ( eth_pkt.dst==ctrl_log_table[i]['MAC'] and ip_pkt.dst==ctrl_log_table[i]['IP'] and tcp_pkt.dst_port==ctrl_log_table[i]['PORT']):
                            self.packet_block = False
                        else:
                            self.packet_block = True
                        i = i + 1

                    self.logger.debug('Block the incoming packet-flow : %s', self.packet_block)
                    if (self.packet_block == True):
                        priority = 1001
                        match = parser.OFPMatch(in_port= affected_port,  eth_type=eth_pkt.ethertype, ip_proto=ip_pkt.proto, ipv4_src=ip_pkt.src, ipv4_dst=ip_pkt.dst, tcp_dst=tcp_pkt.dst_port )
                        actions = [parser.OFPActionOutput(drop_port)]
                        self.add_flow(datapath, priority, match, actions)
                        self.packet_block = False
                        self.total_rule_no = self.total_rule_no + 1
                        self.present_time = time.time()
                        time_diff = self.present_time - self.past_time
                        self.past_time = self.present_time
                        self.logger.debug('Total %s rules are inserted in Switch %s at time %s', self.total_rule_no, dpid, self.present_time)
                        self.logger.debug('delay in processing the request is %s sec', time_diff)

                        results_file = open("results-ruleInsert-standard.pcap", "a")
                        results_file.write("%s %s %s\n" % (self.total_rule_no, self.present_time, time_diff))
                        results_file.close()

# ..........................................................................................................

# ........................................ Cycelic Network Control ......................................

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if not datapath.id in self.datapaths:
                self.logger.debug('register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.debug('deregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            ctrl_DB = dblogger['route_status']
            for datapath in self.datapaths.values():
                dpid = datapath.id
                ofproto = datapath.ofproto
                parser = datapath.ofproto_parser
                self.logger.debug('route_info %s , route_number %s, priority_no %s, routing_path %s', ctrl_DB['route_info'], ctrl_DB['route_no'], self.rule_priority, self.routing_path)

                if ( datapath.id == affected_switch):
                    self.send_aggregate_stats_request(datapath)

                self.ctrl_input_in = time.time()
                if (ctrl_DB['route_info'] == True and ctrl_DB['route_no'] != '0'):
                    self.routing_path = ctrl_DB['route_no']

                    if ( self.Rule_insert_1 == False and self.Rule_insert_2 == False and self.Rule_insert_3 == False and self.Rule_insert_4 == False and self.Rule_insert_5 == False ):
                        self.rule_priority = self.rule_priority + 1
                        ctrl_DB['route_info'] = False
                        dblogger.save(ctrl_DB)
                        self.Rule_insert_1 = True
                        self.Rule_insert_2 = True
                        self.Rule_insert_3 = True
                        self.Rule_insert_4 = True
                        self.Rule_insert_5 = True
                        self.ctrl_input_out = time.time()
                        ctrl_input_diff = self.ctrl_input_out - self.ctrl_input_in
                        self.logger.debug('core-network ctrl-message response-time: %s',ctrl_input_diff)

                        results_file = open("results-ctrldelay-standard.txt", "a")
                        results_file.write("%s\n" % ctrl_input_diff)
                        results_file.close()


                    if ( self.routing_path == '1' ):
                        if ( self.Rule_insert_1 == True and datapath.id == 2 ):
                            priority = self.rule_priority
                            match = parser.OFPMatch(in_port= 1, eth_type=0x0800, ipv4_src='10.0.0.1', ipv4_dst='10.0.0.2')
                            actions = [parser.OFPActionOutput(3)]
                            self.add_flow(datapath, priority, match, actions)
                            priority = self.rule_priority
                            match = parser.OFPMatch(in_port= 3, eth_type=0x0800)
                            actions = [parser.OFPActionOutput(1)]
                            self.add_flow(datapath, priority, match, actions)
                            self.Rule_insert_1 = False
                        if ( self.Rule_insert_2 == True and datapath.id == 1 ):
                            priority = self.rule_priority
                            match = parser.OFPMatch(in_port=1, eth_type=0x0800, ipv4_src='10.0.0.1', ipv4_dst='10.0.0.2')
                            actions = [parser.OFPActionOutput(3)]
                            self.add_flow(datapath, priority, match, actions)
                            priority = self.rule_priority
                            match = parser.OFPMatch(in_port=3, eth_type=0x0800)
                            actions = [parser.OFPActionOutput(1)]
                            self.add_flow(datapath, priority, match, actions)
                            self.Rule_insert_2 = False
                        if ( self.Rule_insert_3 == True and datapath.id == 9 ):
                            priority = self.rule_priority
                            match = parser.OFPMatch(in_port=1, eth_type=0x0800, ipv4_src='10.0.0.1', ipv4_dst='10.0.0.2')
                            actions = [parser.OFPActionOutput(2)]
                            self.add_flow(datapath, priority, match, actions)
                            priority = self.rule_priority
                            match = parser.OFPMatch(in_port=2, eth_type=0x0800)
                            actions = [parser.OFPActionOutput(1)]
                            self.add_flow(datapath, priority, match, actions)
                            self.Rule_insert_3 = False
                        if ( self.Rule_insert_4 == True and datapath.id == 5 ):
                            priority = self.rule_priority
                            match = parser.OFPMatch(in_port=1, eth_type=0x0800, ipv4_src='10.0.0.1', ipv4_dst='10.0.0.2')
                            actions = [parser.OFPActionOutput(2)]
                            self.add_flow(datapath, priority, match, actions)
                            priority = self.rule_priority
                            match = parser.OFPMatch(in_port=2, eth_type=0x0800)
                            actions = [parser.OFPActionOutput(1)]
                            self.add_flow(datapath, priority, match, actions)
                            self.Rule_insert_4 = False
                        if ( self.Rule_insert_5 == True and datapath.id == 6 ):
                            priority = self.rule_priority
                            match = parser.OFPMatch(in_port=3, eth_type=0x0800, ipv4_src='10.0.0.1', ipv4_dst='10.0.0.2')
                            actions = [parser.OFPActionOutput(1)]
                            self.add_flow(datapath, priority, match, actions)
                            priority = self.rule_priority
                            match = parser.OFPMatch(in_port=1, eth_type=0x0800)
                            actions = [parser.OFPActionOutput(3)]
                            self.add_flow(datapath, priority, match, actions)
                            self.Rule_insert_5 = False
                    elif ( self.routing_path == '2' ):
                        if ( self.Rule_insert_1 == True and datapath.id == 2 ):
                            priority = self.rule_priority
                            match = parser.OFPMatch(in_port=1 , eth_type=0x0800, ipv4_src='10.0.0.1', ipv4_dst='10.0.0.2')
                            actions = [parser.OFPActionOutput(2)]
                            self.add_flow(datapath, priority, match, actions)
                            priority = self.rule_priority
                            match = parser.OFPMatch(in_port=2 , eth_type=0x0800)
                            actions = [parser.OFPActionOutput(1)]
                            self.add_flow(datapath, priority, match, actions)
                            self.Rule_insert_1 = False
                        if ( self.Rule_insert_2 == True and datapath.id == 4 ):
                            priority = self.rule_priority
                            match = parser.OFPMatch(in_port=1, eth_type=0x0800, ipv4_src='10.0.0.1', ipv4_dst='10.0.0.2')
                            actions = [parser.OFPActionOutput(3)]
                            self.add_flow(datapath, priority, match, actions)
                            priority = self.rule_priority
                            match = parser.OFPMatch(in_port=3, eth_type=0x0800)
                            actions = [parser.OFPActionOutput(1)]
                            self.add_flow(datapath, priority, match, actions)
                            self.Rule_insert_2 = False
                        if ( self.Rule_insert_3 == True and datapath.id == 8 ):
                            priority = self.rule_priority
                            match = parser.OFPMatch(in_port=1, eth_type=0x0800, ipv4_src='10.0.0.1', ipv4_dst='10.0.0.2')
                            actions = [parser.OFPActionOutput(2)]
                            self.add_flow(datapath, priority, match, actions)
                            priority = self.rule_priority
                            match = parser.OFPMatch(in_port=2, eth_type=0x0800)
                            actions = [parser.OFPActionOutput(1)]
                            self.add_flow(datapath, priority, match, actions)
                            self.Rule_insert_3 = False
                        if ( self.Rule_insert_4 == True and datapath.id == 7 ):
                            priority = self.rule_priority
                            match = parser.OFPMatch(in_port=3, eth_type=0x0800, ipv4_src='10.0.0.1', ipv4_dst='10.0.0.2')
                            actions = [parser.OFPActionOutput(2)]
                            self.add_flow(datapath, priority, match, actions)
                            priority = self.rule_priority
                            match = parser.OFPMatch(in_port=2, eth_type=0x0800)
                            actions = [parser.OFPActionOutput(3)]
                            self.add_flow(datapath, priority, match, actions)
                            self.Rule_insert_4 = False
                        if ( self.Rule_insert_5 == True and datapath.id == 6 ):
                            priority = self.rule_priority
                            match = parser.OFPMatch(in_port=2, eth_type=0x0800, ipv4_src='10.0.0.1', ipv4_dst='10.0.0.2')
                            actions = [parser.OFPActionOutput(1)]
                            self.add_flow(datapath, priority, match, actions)
                            priority = self.rule_priority
                            match = parser.OFPMatch(in_port=1, eth_type=0x0800)
                            actions = [parser.OFPActionOutput(2)]
                            self.add_flow(datapath, priority, match, actions)
                            self.Rule_insert_5 = False
            hub.sleep(1)
# ......................................................................................................


