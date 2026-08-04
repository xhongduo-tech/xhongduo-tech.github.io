---
pageClass: plain-doc
---

# Cryptography & Information Security

Cryptography is the mathematical foundation of information security, while information security is cryptography's engineering realization in systems and networks. This category mirrors the chapter structure of William Stallings' *Cryptography and Network Security: Principles and Practice*, covering a complete knowledge chain from classical ciphers to post-quantum cryptography, from protocols to system security.

## Topic Plan

<ProgressGrid cat="cs/cryptography-security" />


### Part 1 Classical Ciphers and Fundamentals of Cryptanalysis

- [ ] Fundamental concepts of cryptography: plaintext, ciphertext, key, and the five-tuple of a cryptosystem
- [ ] Attack models and threat classification: passive vs. active attacks
- [ ] Kerckhoffs's Principle and the assumptions of modern cipher design
- [ ] Monoalphabetic substitution ciphers: Caesar, affine, and arbitrary monoalphabetic substitution
- [ ] Statistical analysis of monoalphabetic substitution ciphers: letter-frequency attacks
- [ ] Polyalphabetic substitution ciphers: the Vigenère cipher
- [ ] Cryptanalysis of polyalphabetic ciphers: the Kasiski test and index of coincidence
- [ ] Transposition ciphers: the rail fence and columnar transposition

### Part 2 Information-Theoretic Security and the One-Time Pad

- [ ] Shannon's theory of secrecy systems: entropy, conditional entropy, and equivocation
- [ ] Perfect secrecy: definition and criteria
- [ ] The one-time pad and its proof of security
- [ ] Engineering limitations of the one-time pad and the significance of ciphertext-only attacks
- [ ] The distinction between unconditional and computational security

### Part 3 Block Ciphers

- [ ] Design principles of block ciphers: confusion and diffusion
- [ ] The Feistel network structure and its invertibility
- [ ] The overall DES structure: initial permutation, 16 rounds, and inverse permutation
- [ ] The DES round function: expansion, S-boxes, and P-permutation
- [ ] DES security: weak keys, differential and linear cryptanalysis
- [ ] Multiple DES and meet-in-the-middle attacks: why 2DES fails, and the construction of 3DES
- [ ] Mathematical foundations of AES: arithmetic over the finite field GF(2^8)
- [ ] AES round transformations: SubBytes, ShiftRows, MixColumns, and AddRoundKey
- [ ] AES key expansion and security analysis
- [ ] Block cipher modes of operation: ECB and CBC
- [ ] Block cipher modes of operation: CFB, OFB, and CTR
- [ ] Authenticated encryption modes: GCM and CCM
- [ ] Block cipher padding schemes and padding oracle attacks

### Part 4 Stream Ciphers

- [ ] The basic stream cipher model: keystream generators
- [ ] Linear feedback shift registers (LFSRs) and their period properties
- [ ] Combined LFSR generators and nonlinear filtering
- [ ] The RC4 algorithm and its bias attacks
- [ ] The ChaCha20 stream cipher and modern stream cipher design

### Part 5 Number-Theoretic Foundations of Public-Key Cryptography

- [ ] Divisibility, the Euclidean algorithm, and the extended Euclidean algorithm
- [ ] Modular arithmetic and congruence: the ring of residue classes Z_n
- [ ] Fermat's little theorem and Euler's theorem
- [ ] Primality testing: the Miller–Rabin algorithm
- [ ] The discrete logarithm problem and cyclic groups
- [ ] The Chinese Remainder Theorem (CRT) and its applications in cryptography

### Part 6 Public-Key Cryptosystems

- [ ] The conceptual origins of public-key cryptography: one-way and trapdoor one-way functions
- [ ] The RSA algorithm: key generation, encryption, and decryption
- [ ] Proof of correctness and security assumptions of RSA
- [ ] RSA implementation optimizations: CRT acceleration and low-exponent attacks
- [ ] RSA padding schemes: PKCS#1 v1.5 and OAEP
- [ ] The Diffie–Hellman key exchange protocol
- [ ] The ElGamal encryption scheme
- [ ] Elliptic curve fundamentals: elliptic curve groups and the elliptic curve discrete logarithm problem
- [ ] Elliptic curve cryptography (ECC): ECDH key exchange
- [ ] Comparing ECC and RSA: key length and security strength

### Part 7 Hash Functions and Message Authentication

- [ ] Properties of cryptographic hash functions: preimage, second-preimage, and collision resistance
- [ ] Iterated hash structure: the Merkle–Damgård construction
- [ ] Collision attacks on MD5 and SHA-1
- [ ] The SHA-2 family: SHA-256 and SHA-512
- [ ] SHA-3 and the sponge construction
- [ ] The birthday paradox and birthday attacks
- [ ] The principles of message authentication codes (MACs)
- [ ] The construction and security of HMAC
- [ ] Hash-based length extension attacks

### Part 8 Digital Signatures

- [ ] Digital signatures: concepts and security requirements
- [ ] The RSA digital signature scheme
- [ ] The ElGamal digital signature scheme
- [ ] The Digital Signature Algorithm (DSA)
- [ ] The Elliptic Curve Digital Signature Algorithm (ECDSA)
- [ ] EdDSA and Ed25519
- [ ] Signature unforgeability: the EUF-CMA security model

### Part 9 Key Distribution and Public-Key Infrastructure (PKI)

- [ ] The symmetric key distribution problem and key distribution centers (KDCs)
- [ ] The Needham–Schroeder protocol
- [ ] The Kerberos authentication system
- [ ] Public-key certificates and the X.509 standard
- [ ] Certificate chains, trust models, and certificate authorities (CAs)
- [ ] Certificate revocation: CRLs and OCSP

### Part 10 Authentication Protocols

- [ ] Basic authentication factors: passwords, tokens, and biometrics
- [ ] Password storage: salted hashing and slow hash functions (PBKDF2, bcrypt, Argon2)
- [ ] Challenge–response authentication protocols
- [ ] Replay attacks and their defenses: nonces, timestamps, and sequence numbers
- [ ] Man-in-the-middle attacks and authenticated key exchange
- [ ] Multi-factor authentication (MFA) and FIDO2/WebAuthn

### Part 11 TLS/SSL in Depth

- [ ] The history and evolution of SSL/TLS versions
- [ ] The TLS handshake protocol: key and parameter negotiation
- [ ] The TLS record protocol: encryption and integrity protection
- [ ] Differences between TLS 1.2 and TLS 1.3
- [ ] HTTPS and the certificate validation process
- [ ] Notable TLS attacks: BEAST, POODLE, Heartbleed, and downgrade attacks

### Part 12 Network Security

- [ ] Firewalls: packet filtering, stateful inspection, and application-layer gateways
- [ ] Intrusion detection systems (IDS) and intrusion prevention systems (IPS)
- [ ] Denial-of-service (DoS) and distributed denial-of-service (DDoS) attacks
- [ ] VPN fundamentals: the IPSec protocol suite
- [ ] TLS VPNs and WireGuard
- [ ] Wireless network security: WEP's flaws and WPA2/WPA3
- [ ] Email security: PGP and S/MIME

### Part 13 Software Security

- [ ] Buffer overflows: the principles and exploitation of stack overflows
- [ ] Buffer overflow defenses: stack canaries, DEP/NX, and ASLR
- [ ] Format string vulnerabilities and integer overflows
- [ ] SQL injection: principles, exploitation, and parameterized query defenses
- [ ] Cross-site scripting (XSS): reflected, stored, and DOM-based
- [ ] Cross-site request forgery (CSRF) and its defenses
- [ ] Deserialization vulnerabilities and command injection
- [ ] Software supply chain security: dependency confusion, poisoning, and SBOMs

### Part 14 System Security

- [ ] Operating system permission models: users, groups, and access control lists
- [ ] Discretionary access control (DAC) and mandatory access control (MAC)
- [ ] Capabilities and the principle of least privilege
- [ ] Sandboxing mechanisms: chroot, namespaces, and seccomp
- [ ] Trusted computing: TPM and secure boot
- [ ] Malware: viruses, worms, trojans, and ransomware

### Part 15 Privacy-Preserving Technologies

- [ ] Anonymization and k-anonymity: de-anonymization attacks
- [ ] Fundamentals of differential privacy
- [ ] Differential privacy mechanisms: the Laplace and exponential mechanisms
- [ ] Homomorphic encryption: partially and fully homomorphic
- [ ] Secure multi-party computation (MPC) and secret sharing
- [ ] Introduction to zero-knowledge proofs: interactive proofs and the Schnorr protocol

### Part 16 Post-Quantum Cryptography

- [ ] The threat of quantum computing: Shor's and Grover's algorithms
- [ ] An overview of the main technical approaches in post-quantum cryptography
- [ ] Lattice-based cryptography: the LWE problem and Kyber
- [ ] Hash-based signatures: Lamport signatures and SPHINCS+
- [ ] The NIST post-quantum cryptography standardization process
- [ ] Hybrid modes and crypto-agility

> After writing is complete: create a new `xxx.md` in this directory, then change the corresponding item above to `- [x] [标题](./xxx)`.
