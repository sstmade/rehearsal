"""Read/extract single members of a remote ZIP via HTTP Range requests (no full download).

Usage:
  python -I remotezip.py list URL [substring]
  python -I remotezip.py get URL OUTDIR member[=dest_name] [...]
Downloaded bytes are treated as data only; nothing is executed.
"""
import io, os, subprocess, sys, zipfile

UA = {"User-Agent": "rehearsal-datascout/0.1"}


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        out = subprocess.run(["curl", "-sIL", "-A", UA["User-Agent"], "-w", "%{url_effective}", url],
                             capture_output=True, text=True, check=True).stdout
        self.url = out.strip().splitlines()[-1]
        sizes = [l.split(":", 1)[1].strip() for l in out.splitlines() if l.lower().startswith("content-length:")]
        self.size = int(sizes[-1])
        self.pos = 0
        self.cache = {}  # block -> bytes
        self.block = 1 << 20
        self.fetched = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _range(self, a, b):
        data = subprocess.run(["curl", "-sfL", "-A", UA["User-Agent"], "-r", f"{a}-{b}", self.url],
                              capture_output=True, check=True).stdout
        assert len(data) == b - a + 1, (len(data), a, b)
        self.fetched += len(data)
        return data

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if n > 4 * self.block:  # big read: direct
            data = self._range(self.pos, self.pos + n - 1)
        else:
            out = []
            p, end = self.pos, self.pos + n
            while p < end:
                bi = p // self.block
                if bi not in self.cache:
                    a = bi * self.block
                    self.cache[bi] = self._range(a, min(a + self.block, self.size) - 1)
                blk = self.cache[bi]
                off = p - bi * self.block
                take = min(end - p, len(blk) - off)
                out.append(blk[off:off + take])
                p += take
            data = b"".join(out)
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


def open_zip(url):
    f = RangeFile(url)
    return f, zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 16))


if __name__ == "__main__":
    cmd, url = sys.argv[1], sys.argv[2]
    f, z = open_zip(url)
    if cmd == "list":
        sub = sys.argv[3] if len(sys.argv) > 3 else ""
        for i in z.infolist():
            if sub in i.filename:
                print(i.filename, i.file_size)
    elif cmd == "get":
        outdir = sys.argv[3]
        for m in sys.argv[4:]:  # "member" or "member=dest_name"
            m, _, name = m.partition("=")
            dst = os.path.join(outdir, name or os.path.basename(m))
            with z.open(m) as src, open(dst, "wb") as o:
                o.write(src.read())
            print("wrote", dst)
    print(f"[fetched {f.fetched/1e6:.1f} MB of {f.size/1e6:.0f} MB]", file=sys.stderr)
