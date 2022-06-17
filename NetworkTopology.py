
from mininet.net import Mininet
from mininet.node import Controller, RemoteController, OVSKernelSwitch, IVSSwitch, UserSwitch, Ryu, OVSSwitch
from mininet.link import Link, TCLink, Intf
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from subprocess import call
from mininet.node import CPULimitedHost, Host, Node
from mininet.util import dumpNodeConnections


def topology():

    "Create a network."
    net = Mininet(controller=RemoteController, link=TCLink, switch=OVSKernelSwitch)
    info( '*** Adding controller\n' )
    c1=net.addController(name='c1', controller=RemoteController, ip='127.0.0.1',port=6653)  

    info( '*** Adding switches\n' )
    s1 = net.addSwitch( 's1', cls=OVSKernelSwitch, dpid='1', protocols='OpenFlow14', mac='00:00:00:00:00:01')  
    s2 = net.addSwitch( 's2', cls=OVSKernelSwitch, dpid='2', protocols='OpenFlow14', mac='00:00:00:00:00:02')  
    s3 = net.addSwitch( 's3', cls=OVSKernelSwitch, dpid='3', protocols='OpenFlow14', mac='00:00:00:00:00:03')
    s4 = net.addSwitch( 's4', cls=OVSKernelSwitch, dpid='4', protocols='OpenFlow14', mac='00:00:00:00:00:04')  
    s5 = net.addSwitch( 's5', cls=OVSKernelSwitch, dpid='5', protocols='OpenFlow14', mac='00:00:00:00:00:05')  
    s6 = net.addSwitch( 's6', cls=OVSKernelSwitch, dpid='6', protocols='OpenFlow14', mac='00:00:00:00:00:06')  
    s7 = net.addSwitch( 's7', cls=OVSKernelSwitch, dpid='7', protocols='OpenFlow14', mac='00:00:00:00:00:07')  
    s8 = net.addSwitch( 's8', cls=OVSKernelSwitch, dpid='8', protocols='OpenFlow14', mac='00:00:00:00:00:08')  
    s9 = net.addSwitch( 's9', cls=OVSKernelSwitch, dpid='9', protocols='OpenFlow14', mac='00:00:00:00:00:09')
    s10 = net.addSwitch( 's10', cls=OVSKernelSwitch, dpid='10', protocols='OpenFlow14', mac='00:00:00:00:00:10')

    info( '*** Add hosts\n')
    h1 = net.addHost('h1', cls=Host, mac='00:00:00:00:01:01', ip='10.0.0.1/8')         
    h2 = net.addHost('h2', cls=Host, mac='00:00:00:00:02:02', ip='10.0.0.2/8')
    h3 = net.addHost('h3', cls=Host, mac='00:00:00:00:03:03', ip='10.0.0.3/8')
    h4 = net.addHost('h4', cls=Host, mac='00:00:00:00:04:04', ip='10.0.0.4/8')



    info( '*** Add links\n')
    net.addLink(s1, s2, port1=1, port2=3, cls=TCLink)
    net.addLink(s1, s3, port1=2, port2=3, cls=TCLink)
    net.addLink(s1, s9, port1=3, port2=1, cls=TCLink)

    net.addLink(h1, s2, port1=0, port2=1, cls=TCLink)
    net.addLink(s2, s4, port1=2, port2=1, cls=TCLink)

    net.addLink(h3, s3, port1=0, port2=1, cls=TCLink)
    net.addLink(h4, s3, port1=0, port2=4, cls=TCLink)
    net.addLink(s3, s4, port1=2, port2=2, cls=TCLink)

    net.addLink(s4, s8, port1=3, port2=1, cls=TCLink)

    net.addLink(s5, s6, port1=2, port2=3, cls=TCLink)
    net.addLink(s5, s10, port1=3, port2=2, cls=TCLink)
    net.addLink(s5, s9, port1=1, port2=2, cls=TCLink)

    net.addLink(h2, s6, port1=0, port2=1, cls=TCLink)
    net.addLink(s6, s7, port1=2, port2=2, cls=TCLink)

    net.addLink(s7, s10, port1=1, port2=1, cls=TCLink)
    net.addLink(s7, s8, port1=3, port2=2, cls=TCLink)

    net.build()	
    info( '*** Starting controllers\n')
    c1.start()

    info( '*** Starting switches\n')
    s1.start( [c1] )
    s2.start( [c1] )
    s3.start( [c1] )   
    s4.start( [c1] )
    s5.start( [c1] )
    s6.start( [c1] )   
    s7.start( [c1] )
    s8.start( [c1] )
    s9.start( [c1] )   
    s10.start( [c1] )

    print( "*** Running CLI")
    CLI( net )

    print( "*** Stopping network")
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    topology()

