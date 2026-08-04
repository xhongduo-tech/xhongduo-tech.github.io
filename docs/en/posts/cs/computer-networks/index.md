---
pageClass: plain-doc
---

# Computer Networks

A complete chapter-by-chapter system aligned with Xie Xiren's *Computer Networks* and Tanenbaum's *Computer Networks*, written through from architecture to modern topics.

## Topic Plan

<ProgressGrid cat="cs/computer-networks" />


### Chapter 1 Overview

- [ ] The role of computer networks in the information age
- [ ] Overview of the Internet: a network of networks
- [ ] Composition of the Internet: edge and core parts
- [ ] Circuit switching, message switching, and packet switching
- [ ] Definition and classification of computer networks
- [ ] Performance metrics of computer networks: rate, bandwidth, throughput
- [ ] Delay, delay-bandwidth product, round-trip time, and utilization
- [ ] Computer network architecture: protocols and layering
- [ ] The seven-layer OSI reference model
- [ ] The four-layer TCP/IP architecture
- [ ] The five-layer protocol architecture with encapsulation and decapsulation
- [ ] Entities, protocols, services, and service access points

### Chapter 2 Physical Layer

- [ ] Basic concepts and four main characteristics of the physical layer
- [ ] Fundamentals of data communication: channels, signals, and symbols
- [ ] The limiting capacity of a channel: Nyquist's criterion and Shannon's formula
- [ ] Transmission media: twisted pair, coaxial cable, and optical fiber
- [ ] Wireless transmission: radio, microwave, infrared, and satellite communication
- [ ] Digital modulation: ASK, FSK, PSK, and quadrature amplitude modulation (QAM)
- [ ] Encoding of analog and digital signals: NRZ, Manchester, and differential Manchester
- [ ] Channel multiplexing: frequency-division multiplexing (FDM) and time-division multiplexing (TDM)
- [ ] Statistical time-division multiplexing (STDM)
- [ ] Wavelength-division multiplexing (WDM) and dense wavelength-division multiplexing (DWDM)
- [ ] Code-division multiplexing (CDM) and code-division multiple access (CDMA)
- [ ] Broadband access technologies: ADSL and HFC
- [ ] FTTx fiber access

### Chapter 3 Data Link Layer

- [ ] Functions of the data link layer and point-to-point channels
- [ ] Framing and frame delimitation
- [ ] Transparent transmission: byte stuffing and bit stuffing
- [ ] Error detection: parity checking and cyclic redundancy check (CRC)
- [ ] Stop-and-wait protocol
- [ ] Continuous ARQ protocols and sliding windows
- [ ] Go-Back-N (GBN) and Selective Repeat (SR)
- [ ] Composition and frame format of the Point-to-Point Protocol (PPP)
- [ ] PPP working states and authentication
- [ ] The data link layer over broadcast channels: LAN overview
- [ ] The CSMA/CD protocol: carrier sensing, collision detection, and the backoff algorithm
- [ ] Ethernet channel utilization and the minimum frame length
- [ ] The Ethernet MAC layer: MAC addresses and frame format
- [ ] Extending Ethernet at the physical layer: hubs and collision domains
- [ ] Extending Ethernet at the data link layer: bridges and switches
- [ ] Self-learning and forwarding in Ethernet switches
- [ ] Virtual LANs (VLANs) and 802.1Q
- [ ] High-speed Ethernet: 100BASE-T, Gigabit, and 10-Gigabit Ethernet

### Chapter 4 Network Layer

- [ ] Two services provided by the network layer: virtual circuits and datagrams
- [ ] Internet Protocol (IP) and the virtual internet
- [ ] IPv4 addresses: classful addressing and dotted-decimal notation
- [ ] The relationship between IP addresses and hardware (MAC) addresses
- [ ] Subnetting and subnet masks
- [ ] Forwarding of packets with subnetting
- [ ] Classless addressing (CIDR) and longest prefix matching
- [ ] IPv4 datagram format and fragmentation
- [ ] Address Resolution Protocol (ARP)
- [ ] Internet Control Message Protocol (ICMP): ping and traceroute
- [ ] Routing on the Internet: static routing and dynamic routing
- [ ] Autonomous systems with interior gateway protocols and exterior gateway protocols
- [ ] Routing Information Protocol (RIP): the distance-vector algorithm
- [ ] Open Shortest Path First (OSPF): the link-state algorithm
- [ ] Border Gateway Protocol (BGP): path vectors and routing policies
- [ ] Router structure and the packet forwarding process
- [ ] The basic header of IPv6 and address representation
- [ ] Transition from IPv4 to IPv6: tunneling and dual stack
- [ ] Network Address Translation (NAT)

### Chapter 5 Transport Layer

- [ ] Overview of transport layer protocols: communication between processes
- [ ] Ports, sockets, and multiplexing/demultiplexing
- [ ] User Datagram Protocol (UDP): header and characteristics
- [ ] Overview of the Transmission Control Protocol (TCP): a connection-oriented byte stream
- [ ] The header format of a TCP segment
- [ ] How reliable transmission works: acknowledgments and retransmission
- [ ] TCP numbering, acknowledgments, and cumulative acknowledgments
- [ ] Choosing the retransmission timeout: RTT estimation and Karn's algorithm
- [ ] Establishing a TCP connection: the three-way handshake
- [ ] Releasing a TCP connection: the four-way handshake
- [ ] The TCP finite state machine
- [ ] Flow control using a sliding window
- [ ] TCP congestion control: slow start and congestion avoidance
- [ ] Fast retransmit and fast recovery
- [ ] Active queue management (AQM) and RED

### Chapter 6 Application Layer

- [ ] Application layer protocols and the process communication model: C/S and P2P
- [ ] Domain Name System (DNS): domain name structure and name servers
- [ ] The DNS resolution process: recursive queries and iterative queries
- [ ] File Transfer Protocol (FTP) and TFTP
- [ ] Overview of the World Wide Web (WWW): URLs and the Web architecture
- [ ] Hypertext Transfer Protocol (HTTP): message structure and request methods
- [ ] HTTP persistent connections, cookies, and session management
- [ ] Web caching and proxy servers
- [ ] Overview of the email system: user agents and mail servers
- [ ] Simple Mail Transfer Protocol (SMTP)
- [ ] Mail reading protocols (POP3 and IMAP)
- [ ] Dynamic Host Configuration Protocol (DHCP)
- [ ] P2P-based applications: file distribution and BitTorrent

### Chapter 7 Network Security

- [ ] Overview of network security issues: threat models and types of attacks
- [ ] Two classes of cryptosystems: symmetric-key and public-key cryptography
- [ ] Symmetric-key ciphers: DES and AES
- [ ] Public-key cryptography: RSA and elliptic-curve cryptography (ECC)
- [ ] Digital signatures and message authentication
- [ ] Cryptographic hash functions: MD5, SHA, and integrity checking
- [ ] Key distribution and the Public Key Infrastructure (PKI)
- [ ] Digital certificates and certificate authorities (CAs)
- [ ] Transport layer security: the TLS/SSL handshake and record protocols
- [ ] Firewalls: packet filtering and stateful inspection
- [ ] Intrusion detection systems (IDS) and intrusion prevention systems (IPS)

### Chapter 8 Wireless and Mobile Networks

- [ ] Composition of wireless LANs (WLANs) and the 802.11 architecture
- [ ] The 802.11 MAC layer: CSMA/CA and RTS/CTS
- [ ] The 802.11 frame format and the association process
- [ ] Wireless personal area networks: Bluetooth
- [ ] Cellular mobile networks: the evolution of GSM, LTE, and 5G
- [ ] Mobility management in cellular networks: handover and roaming
- [ ] The impact of wireless networks on higher-layer protocols

### Chapter 9 Modern Networking Topics

- [ ] HTTP/2: binary framing, multiplexing, and header compression (HPACK)
- [ ] HTTP/3 and QUIC: reliable transport over UDP
- [ ] How content delivery networks (CDNs) work
- [ ] Software-defined networking (SDN) and OpenFlow
- [ ] Data center networks and network functions virtualization (NFV)

> After finishing a section: create `xxx.md` in this directory, then change the corresponding item above to `- [x] [Title](./xxx)`.
