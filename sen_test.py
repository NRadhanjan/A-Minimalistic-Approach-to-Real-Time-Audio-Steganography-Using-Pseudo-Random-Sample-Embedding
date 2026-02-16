# sen_test.py
import socket, struct, numpy as np, random, time, math

# Config
CHUNK = 1024
RATE = 44100
BITS_PER_FRAME = 32
PRNG_SEED = 12345
HOST = "127.0.0.1"
PORT = 5000

def text_to_bits(s):
    bits = []
    for ch in s:
        b = format(ord(ch), '08b')
        bits.extend(list(b))
    return bits

def main():
    msg = input("Enter secret test message: ")
    bits = text_to_bits(msg)
    total_bits = len(bits)
    print(f"[Sender-Test] Message length: {len(msg)} chars, {total_bits} bits")

    # (5.2) Calculate embedding capacity
    embedding_rate_bps = (BITS_PER_FRAME * RATE) / CHUNK
    print(f"[5.2] Embedding Capacity = {embedding_rate_bps:.2f} bits/sec")

    # Connect
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    hdr = struct.pack(">I I H", PRNG_SEED, total_bits, BITS_PER_FRAME)
    s.sendall(hdr)
    print("[Sender-Test] Header sent.")

    # Simulate 200 frames of random "audio"
    frame_id = 0
    bit_ptr = 0
    latency_records = []
    snr_records = []

    while frame_id < 200:
        # Simulate input audio samples
        original = np.random.randint(-32768, 32767, CHUNK, dtype=np.int16)
        data = original.tobytes()
        t_send = time.time()  # timestamp before sending

        # Send timestamp first (float, 8 bytes)
        s.sendall(struct.pack(">d", t_send))

        # Embed bits
        k = min(BITS_PER_FRAME, total_bits - bit_ptr)
        mod = original.copy()
        if k > 0:
            rnd = random.Random(PRNG_SEED ^ frame_id)
            indices = rnd.sample(range(CHUNK), k)
            for i, idx in enumerate(indices):
                bit = int(bits[bit_ptr + i])
                mod[idx] = (mod[idx] & ~1) | bit
            bit_ptr += k

        # Compute (5.3) SNR metric
        diff = original.astype(np.int32) - mod.astype(np.int32)
        mse = np.sum(diff ** 2)
        sig = np.sum(original.astype(np.int32) ** 2)
        snr = 10 * math.log10(sig / (mse + 1e-9))
        snr_records.append(snr)

        # Send stego audio
        s.sendall(mod.tobytes())

        if frame_id % 50 == 0:
            avg_snr = np.mean(snr_records) if snr_records else 0
            print(f"[5.3] Avg SNR @ frame {frame_id}: {avg_snr:.2f} dB")

        frame_id += 1

    print("[Sender-Test] All bits sent.")
    s.close()

    # Summary
    print(f"[5.3] Overall Average SNR: {np.mean(snr_records):.2f} dB")
    print("[Sender-Test] Connection closed.")

if __name__ == "__main__":
    main()
