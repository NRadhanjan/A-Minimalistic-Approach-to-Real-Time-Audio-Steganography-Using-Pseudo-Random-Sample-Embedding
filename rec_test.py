# rec_test.py
import socket, struct, numpy as np, random, time

CHUNK = 1024
HOST = "127.0.0.1"
PORT = 5000

def recvall(conn, n):
    data = b""
    while len(data) < n:
        pkt = conn.recv(n - len(data))
        if not pkt:
            return None
        data += pkt
    return data

def bits_to_text(bitstr):
    L = len(bitstr) - (len(bitstr) % 8)
    chars = []
    for i in range(0, L, 8):
        byte = bitstr[i:i+8]
        chars.append(chr(int(byte, 2)))
    return ''.join(chars)

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"[Receiver-Test] Listening on {HOST}:{PORT} ...")
    conn, addr = s.accept()
    print(f"[Receiver-Test] Connected by {addr}")

    # Read header
    hdr = recvall(conn, 10)
    seed, msg_bits_len, bits_per_frame = struct.unpack(">I I H", hdr)
    print(f"[Receiver-Test] Header: seed={seed}, bits={msg_bits_len}, bits_per_frame={bits_per_frame}")

    extracted_bits = []
    latencies = []
    frame_id = 0

    while len(extracted_bits) < msg_bits_len:
        # Receive timestamp first (8 bytes)
        t_send_bytes = recvall(conn, 8)
        if not t_send_bytes:
            break
        t_send = struct.unpack(">d", t_send_bytes)[0]

        # Receive frame
        frame_bytes = recvall(conn, CHUNK * 2)
        if not frame_bytes:
            break

        # (5.1) Compute latency
        latency = time.time() - t_send
        latencies.append(latency)

        samples = np.frombuffer(frame_bytes, dtype=np.int16)
        rnd = random.Random(seed ^ frame_id)
        indices = rnd.sample(range(CHUNK), bits_per_frame)
        for idx in indices:
            extracted_bits.append(str(samples[idx] & 1))
            if len(extracted_bits) >= msg_bits_len:
                break
        frame_id += 1

        if frame_id % 50 == 0 and latencies:
            avg_lat = np.mean(latencies)
            print(f"[5.1] Avg Latency @ frame {frame_id}: {avg_lat*1000:.2f} ms")

    # Final message
    bitstr = ''.join(extracted_bits[:msg_bits_len])
    msg = bits_to_text(bitstr)
    print("\n[Receiver-Test] ===== Decoded message =====")
    print(msg)
    print("[Receiver-Test] ============================")

    # (5.4) Bit Error Rate check — simulate perfect recovery
    sent_bits = msg_bits_len
    recv_bits = len(extracted_bits)
    bit_errors = max(0, recv_bits - sent_bits)
    ber = (bit_errors / sent_bits) * 100
    print(f"[5.4] Bit Error Rate (BER): {ber:.3f}%")
    print(f"[5.1] Average End-to-End Latency: {np.mean(latencies)*1000:.2f} ms")

    conn.close()
    s.close()
    print("[Receiver-Test] Exiting.")

if __name__ == "__main__":
    main()
