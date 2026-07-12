# A Minimalistic Approach to Real-Time Audio Steganography Using Pseudo-Random Sample Embedding
---

## Abstract

Traditional digital steganography operates almost exclusively on pre-recorded, static media files, which makes it fundamentally unsuited to the demands of modern, real-time communication. This project addresses that gap by proposing and implementing a minimal Real-Time Dynamic Audio Steganography Framework that establishes a synchronized, low-latency covert channel within a live audio stream transmitted over TCP/IP.

The system consists of a Transmitter and a Receiver. The Transmitter captures live 44.1 kHz PCM audio, segments it into 1024-sample frames, and uses a Pseudo-Random Number Generator (PRNG) seeded with a shared key to deterministically select sample indices for Least Significant Bit (LSB) embedding. This PRNG-based indexing keeps the sender and receiver perfectly aligned without any explicit signaling, which maximizes stealth and minimizes delay.

Experimental evaluation of the prototype confirmed its feasibility and performance:

- Average end-to-end latency of approximately 10.87 milliseconds
- Maximum theoretical embedding capacity of 1378.12 bits per second
- Average Signal-to-Noise Ratio (SNR) of 103.33 dB, indicating negligible acoustic distortion
- Bit Error Rate (BER) of 0.000 percent, confirming complete message integrity

These results demonstrate that dynamic, PRNG-synchronized LSB embedding can support practical, high-fidelity covert communication in live audio streams, and they establish a foundation for future work on adaptive masking and stronger cryptographic protection.

---

## Table of Contents

1. Introduction
2. Motivation and Problem Statement
3. Related Work
4. System Architecture
5. Methodology
6. Installation
7. Usage
8. Configuration Parameters
9. Limitations
10. Future Work
    
---

## 1. Introduction

Encryption secures the content of a message, but the presence of overtly encrypted traffic itself often draws attention from adversaries or automated traffic analysis tools. Steganography addresses this problem differently: it conceals the existence of a message altogether by embedding it within ordinary, innocuous carrier media such as images, video, or audio.

Audio is a particularly effective carrier because subtle modifications to a waveform can exploit the redundancy and psychoacoustic masking properties of human hearing, making them imperceptible under normal listening conditions. However, the majority of existing audio steganography research targets offline, file-based scenarios, embedding data into audio before it is stored or transmitted. This precludes such methods from supporting real-time, continuous covert communication, which is increasingly relevant for modern networked applications.

This project develops a minimal real-time audio steganography system that transmits hidden messages continuously between two systems over a local network, converting a traditionally passive steganographic technique into an active, dynamic covert communication channel.

---

## 2. Motivation and Problem Statement

The core technical challenge in real-time steganography is synchronization: the sender and receiver must agree on exactly which samples in a continuous audio stream carry hidden data, without exchanging metadata that would itself reveal the presence of a covert channel or introduce additional latency.

This project addresses that challenge using LSB modification, chosen for its computational simplicity and minimal perceptual impact, combined with a PRNG seeded from a shared secret key. The PRNG deterministically generates the same sequence of embedding indices on both ends, allowing the receiver to mirror the transmitter's embedding process exactly, with no handshake or signaling overhead.

The design goals for the prototype were threefold:

- Maintain imperceptibility of the stego-audio relative to the original signal
- Ensure seamless real-time synchronization between transmitter and receiver
- Minimize end-to-end latency to remain below the threshold of perceptible delay

---

## 3. Related Work

Digital steganography has evolved substantially from early LSB-based image and audio techniques toward more sophisticated multimedia approaches. Prior audio steganography research has explored phase coding, echo hiding, and spread spectrum techniques to improve robustness against compression and signal processing attacks.

A smaller body of work has examined real-time and streaming steganography, including PRNG-based embedding schemes for live audio that use synchronized key sequences to align sender and receiver without additional signaling. However, existing implementations in this space commonly suffer from high buffering delay or require complex synchronization protocols, which limits their real-time feasibility.

More recent research has begun integrating encryption and psychoacoustic modeling to strengthen both the security and perceptual transparency of steganographic systems, along with early exploration of deep learning and generative adversarial network (GAN) based embedding strategies. This project builds on the PRNG-synchronization line of research and contributes a lightweight, practical prototype demonstrating live covert communication using LSB-PRNG embedding combined with AES encryption.

---

## 4. System Architecture

The framework implements a full-duplex real-time audio steganography pipeline across two endpoints connected by a dedicated TCP socket stream: a Transmitter and a Receiver.

```
                              TCP/IP Socket Stream
    ------------------------                                 ------------------------
   |      TRANSMITTER       |  ----- stego-audio ------->   |       RECEIVER         |
   |                        |         frames                |                        |
   |  Audio Capture         |                               |  Network Reception     |
   |  (PyAudio, 44.1 kHz,   |                               |                        |
   |   16-bit PCM, 1024-    |                               |  Dynamic Extraction    |
   |   sample frames)       |                               |  (PRNG + adaptive      |
   |                        |                               |   depth rules)         |
   |  AES-CBC Encryption    |                               |                        |
   |                        |                               |  AES-CBC Decryption    |
   |  Dynamic LSB Embedding |                               |                        |
   |  (PRNG-indexed,        |                               |  Message               |
   |   energy-adaptive)     |                               |  Reconstruction        |
   |                        |                               |                        |
   |  Network Transmission  |                               |  Audio Playback        |
   ------------------------                                  ------------------------
```

The four core modules present on both ends are: Audio Capture, Encryption, Dynamic Embedding/Extraction, and Network Transmission/Reception. Each is optimized for low-latency, frame-by-frame operation rather than batch processing.

---

## 5. Methodology

### 5.1 Transmitter Pipeline

The transmitter captures live audio from the system microphone using PyAudio with the following parameters:

- Sampling rate: 44.1 kHz
- Sample depth: 16-bit PCM
- Frame size: 1024 samples (approximately 23 milliseconds)

These parameters balance latency against audio fidelity. Each frame is passed directly into the embedding pipeline without additional buffering.

Before embedding, the plaintext message is encrypted using AES in Cipher Block Chaining (CBC) mode. The message is converted to bytes, padded to 16-byte blocks using PKCS7 padding, encrypted with a shared symmetric key and initialization vector, and serialized into a bitstream.

```
Algorithm 1: AES-Based Message Encryption
Input:  plaintext M, key K, initialization vector IV
Output: encrypted bitstream B_enc

1: Convert M to a byte sequence
2: Pad to 16-byte blocks (PKCS7)
3: C <- AES_Encrypt_CBC(M, K, IV)
4: B_enc <- ConvertBytesToBits(C)
5: return B_enc
```

The embedding stage uses a dynamic, energy-adaptive LSB strategy. High-energy (loud or complex) frames permit deeper bit modification, up to the third LSB, while low-energy frames are restricted to the first LSB or skipped entirely to avoid audible distortion. A PRNG seeded with a pre-shared key determines the exact sample indices used for embedding in each frame.

```
Algorithm 2: Dynamic LSB Embedding
Input:  frame F[1..N], bitstream B, PRNG seed S
Output: stego frame F'

1:  E <- Energy(F)
2:  if E > threshold_high then depth <- 3
3:  else if E > threshold_low then depth <- 2
4:  else depth <- 1
5:  indices <- PRNG(S).sample(N, |B|)
6:  for i in 1..|B| do
7:      bit_pos <- depth(i)
8:      F'[indices[i]] <- SetLSB(F[indices[i]], B[i], bit_pos)
9:  return F'
```

Stego-frames are transmitted over TCP, chosen for its ordered, reliable delivery guarantees. Each frame is serialized into bytes and sent immediately after embedding to maintain continuous streaming with negligible buffering delay.

### 5.2 Receiver Pipeline

The receiver mirrors the transmitter pipeline in reverse:

1. The network reception component reads incoming byte streams.
2. The dynamic extraction module identifies modified samples using the same PRNG sequence and adaptive depth rules as the transmitter.
3. Extracted ciphertext bits are reassembled and decrypted using the shared AES key and initialization vector to reconstruct the plaintext message.

```
Algorithm 3: Dynamic Extraction and Decryption
Input:  stego frame F', PRNG seed S, key K, IV
Output: plaintext M

1: indices <- PRNG(S).sample(N)
2: bits <- ExtractLSBs(F', indices)
3: C <- ConvertBitsToBytes(bits)
4: M <- AES_Decrypt_CBC(C, K, IV)
5: return M
```

To preserve a natural listening experience, the receiver simultaneously forwards raw stego-frames to the system audio output buffer. Playback runs in parallel with extraction, adding less than 50 milliseconds of additional delay.

### 5.3 Control Flow Summary

Transmitter: capture live audio, encrypt the message, dynamically embed bits, stream via TCP.

Receiver: receive frames, extract bits using the synchronized PRNG, decrypt, play audio.

This pipeline maintains imperceptibility, synchronization, and low latency throughout continuous, live operation.

---

## 6. Installation

### Prerequisites

- Python 3.x
- PyAudio, for microphone capture and audio playback
- A cryptography library supporting AES-CBC (for example, PyCryptodome or the `cryptography` package)
- Two machines (or two processes) on the same local network, or a loopback setup on a single machine for testing

### Setup

```bash
git clone https://github.com/NRadhanjan/A-Minimalistic-Approach-to-Real-Time-Audio-Steganography-Using-Pseudo-Random-Sample-Embedding.git
cd A-Minimalistic-Approach-to-Real-Time-Audio-Steganography-Using-Pseudo-Random-Sample-Embedding
pip install -r requirements.txt
```

Note: PyAudio depends on PortAudio. On Debian/Ubuntu systems, install the system dependency first if the pip install fails:

```bash
sudo apt-get install portaudio19-dev
pip install pyaudio
```

---

## 7. Usage

Start the receiver first so that it is listening for an incoming connection, then start the transmitter.

```bash
# On the receiving machine
python receiver.py

# On the transmitting machine
python transmitter.py
```

The transmitter will prompt for a secret message. The message is encrypted, embedded into the live audio stream using PRNG-selected sample indices, and transmitted over TCP. The receiver plays the incoming audio in real time while simultaneously extracting and decrypting the hidden message, then displays the reconstructed plaintext.

Both endpoints must be configured with the same shared PRNG seed, AES key, and initialization vector for successful synchronization and decryption.

---

## 8. Configuration Parameters

| Parameter | Default Value | Description |
|---|---|---|
| Sampling rate | 44.1 kHz | Audio capture and playback rate |
| Sample depth | 16-bit PCM | Bit depth of captured audio samples |
| Frame size | 1024 samples (~23 ms) | Unit of processing for capture, embedding, and transmission |
| Embedding depth | 1st to 3rd LSB | Selected dynamically based on frame energy |
| Transport protocol | TCP | Used for ordered, reliable stego-frame delivery |
| Encryption | AES-CBC | Applied to the message prior to embedding |

Update this table to reflect the actual configurable values and their locations in the codebase (for example, constants at the top of `transmitter.py` and `receiver.py`).

---

## 9. Limitations

The current prototype is intentionally minimal and does not yet include several features that would be necessary for production or adversarial deployment:

- No forward error correction, so packet loss or network noise can degrade message recovery
- No psychoacoustic modeling beyond simple frame-energy thresholds for adaptive depth selection
- Assumes a stable, low-loss TCP connection between transmitter and receiver
- The shared PRNG seed, AES key, and initialization vector must currently be pre-shared out of band; there is no key exchange protocol implemented
- Evaluated primarily on short test messages under controlled network conditions

---

## 10. Future Work

Planned enhancements to the framework include:

- Upgrading to AES-256 encryption and introducing a proper key exchange or key management protocol
- Incorporating psychoacoustic adaptive masking models to more precisely tune embedding depth and further optimize the balance between capacity and imperceptibility
- Adding forward error correction (FEC) to improve robustness against packet loss and network disturbances
- Exploring deep learning and generative adversarial network (GAN) based embedding strategies for further gains in imperceptibility and capacity

---
