#!/usr/bin/env python3
"""Run commands inside Hexagon Linux under qemu-system-hexagon.

The guest is a buildroot image (vmlinux + rootfs.ext2 from the codelinaro
toolchain artifacts).  Networking is unreliable under system emulation, so
files are moved in and out of the guest through a second ext2 "payload"
disk (/dev/vdb), built on the host with mke2fs -d and read back with
debugfs.  The serial console (UART) drives the interaction via pexpect.

One-shot CLI:

    python3 guest_run.py [--file HOSTPATH ...] [--timeout SECS] -- CMD...

Files given with --file land in /payload (the mounted payload disk); CMD
runs with /payload as the working directory.  Guest stdout/stderr is
streamed to our stdout and also captured with the exit status.  Exits
with the guest command's exit code.

Persistent use:

    sess = GuestSession(files=[...]); sess.boot()
    rc, out = sess.run("./targetnopstandalone-c 100")
    sess.shutdown()

Environment overrides: HEXAGON_TOOLCHAIN, HEXAGON_QEMU_SYSTEM,
HEXAGON_VMLINUX, HEXAGON_ROOTFS.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

import pexpect

HOME = os.path.expanduser("~")
TOOLCHAIN = os.environ.get(
    "HEXAGON_TOOLCHAIN",
    "/opt/clang+llvm-22.1.8-cross-hexagon-unknown-linux-musl/x86_64-ubuntu-22.04")
QEMU = os.environ.get(
    "HEXAGON_QEMU_SYSTEM", os.path.join(TOOLCHAIN, "bin", "qemu-system-hexagon"))
VMLINUX = os.environ.get(
    "HEXAGON_VMLINUX", os.path.join(HOME, ".cache", "hexagon-linux", "vmlinux"))
ROOTFS = os.environ.get(
    "HEXAGON_ROOTFS", os.path.join(HOME, ".cache", "hexagon-linux", "rootfs.ext2"))

MKE2FS = "/usr/sbin/mke2fs"
DEBUGFS = "/usr/sbin/debugfs"

PROMPT = "GUESTSH> "
SENTINEL = "___RC="


def build_payload_image(image_path, files, size_slack_mb=64):
    """Build an ext2 image containing *files* at its root."""
    stage = tempfile.mkdtemp(prefix="hexpayload-")
    try:
        total = 0
        for f in files:
            dest = os.path.join(stage, os.path.basename(f))
            shutil.copy2(f, dest)
            total += os.path.getsize(f)
        size_mb = max(16, total // (1024 * 1024) + size_slack_mb)
        if os.path.exists(image_path):
            os.unlink(image_path)
        subprocess.run(
            [MKE2FS, "-q", "-t", "ext2", "-d", stage, "-N", "2048",
             image_path, "%dm" % size_mb],
            check=True, capture_output=True)
    finally:
        shutil.rmtree(stage)


def extract_from_payload(image_path, guest_path, host_path):
    """Dump a file from the payload image (path relative to its root)."""
    r = subprocess.run(
        [DEBUGFS, "-R", "dump /%s %s" % (guest_path, host_path), image_path],
        capture_output=True, text=True)
    return os.path.exists(host_path) and os.path.getsize(host_path) >= 0 \
        and "File not found" not in r.stderr


class GuestSession(object):
    def __init__(self, files=None, memory="4G", echo_console=True,
                 boot_timeout=300):
        self.files = list(files or [])
        self.memory = memory
        self.echo_console = echo_console
        self.boot_timeout = boot_timeout
        self.child = None
        self.workdir = tempfile.mkdtemp(prefix="hexguest-")
        self.payload_img = os.path.join(self.workdir, "payload.ext2")
        self.console_log = os.path.join(self.workdir, "console.log")

    def boot(self):
        build_payload_image(self.payload_img, self.files)
        cmd = [
            QEMU, "-M", "virt", "-kernel", VMLINUX,
            "-drive", "if=none,id=hd0,file=%s,format=raw,snapshot=on" % ROOTFS,
            "-device", "virtio-blk-device,drive=hd0",
            "-drive", "if=none,id=hd1,file=%s,format=raw" % self.payload_img,
            "-device", "virtio-blk-device,drive=hd1",
            "-m", self.memory, "-accel", "tcg,thread=multi", "-nographic",
        ]
        self.child = pexpect.spawn(cmd[0], cmd[1:], encoding="utf-8",
                                   codec_errors="replace",
                                   timeout=self.boot_timeout,
                                   maxread=65536)
        self._logf = open(self.console_log, "w")
        if self.echo_console:
            class Tee(object):
                def __init__(self, *streams):
                    self.streams = streams
                def write(self, data):
                    for s in self.streams:
                        s.write(data)
                def flush(self):
                    for s in self.streams:
                        s.flush()
            self.child.logfile_read = Tee(self._logf, sys.stdout)
        else:
            self.child.logfile_read = self._logf
        self.child.expect("login:", timeout=self.boot_timeout)
        self.child.sendline("root")
        self.child.expect("# ", timeout=30)
        # Unambiguous prompt, set via concatenation so the echoed command
        # never contains the literal prompt; then disable echo entirely.
        self.child.sendline("stty -echo; export PS1='%s''%s'"
                            % (PROMPT[:4], PROMPT[4:]))
        self.child.expect(PROMPT, timeout=15)
        self._sh("mkdir -p /payload && mount -t ext2 /dev/vdb /payload")
        self._sh("cd /payload")

    def _sh(self, cmdline, timeout=30):
        self.child.sendline(cmdline)
        self.child.expect(PROMPT, timeout=timeout)
        return self.child.before

    def run(self, cmdline, timeout=600):
        """Run cmdline in /payload; return (exit_status, output_text)."""
        wrapped = "{ %s ; } > /payload/last_output.txt 2>&1; " \
                  "rc=$?; cat /payload/last_output.txt; " \
                  "echo; echo %s$rc" % (cmdline, SENTINEL)
        self.child.sendline(wrapped)
        self.child.expect(r"%s(\d+)" % SENTINEL, timeout=timeout)
        rc = int(self.child.match.group(1))
        out = self.child.before
        self.child.expect(PROMPT, timeout=30)
        return rc, out.strip("\r\n")

    def get_file(self, guest_relpath, host_path):
        """After shutdown, extract a file the guest wrote to /payload."""
        return extract_from_payload(self.payload_img, guest_relpath, host_path)

    def shutdown(self):
        if self.child is None:
            return
        try:
            self._sh("cd /; umount /payload || true", timeout=30)
            self.child.sendline("poweroff -f")
            self.child.expect([pexpect.EOF, "reboot: Power down"], timeout=60)
        except Exception:
            pass
        try:
            self.child.close(force=True)
        except Exception:
            pass
        self.child = None
        try:
            self._logf.close()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--file", action="append", default=[],
                   help="host file to place in /payload (repeatable)")
    p.add_argument("--timeout", type=int, default=600,
                   help="seconds to allow the command to run")
    p.add_argument("--quiet", action="store_true",
                   help="do not echo guest console to stdout")
    p.add_argument("cmd", nargs=argparse.REMAINDER,
                   help="command to run in the guest (prefix with --)")
    args = p.parse_args()
    if not args.cmd:
        p.error("no command given")
    cmd = args.cmd
    if cmd[0] == "--":
        cmd = cmd[1:]
    cmdline = " ".join(cmd)

    sess = GuestSession(files=args.file, echo_console=not args.quiet)
    t0 = time.time()
    try:
        sess.boot()
        print("\n[guest_run] booted in %.1fs, running: %s"
              % (time.time() - t0, cmdline), file=sys.stderr)
        rc, out = sess.run(cmdline, timeout=args.timeout)
    finally:
        sess.shutdown()
    if args.quiet:
        sys.stdout.write(out)
    print("\n[guest_run] exit status: %d (console log: %s)"
          % (rc, sess.console_log), file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
