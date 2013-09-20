"""
    Redirect stdout / stderr to temp file

 Originally by R. Kern, 2005
 Adapted by R. Clewley, 2006


Copyright (c) 2005 Robert Kern.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of the Enthought nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

"""

import os
import sys
import tempfile

STDOUT = 1
STDERR = 2

class Redirector(object):
    def __init__(self, fd=STDOUT):
        self.fd = fd
        self.started = False

    def start(self):
        if not self.started:
            self.tmpfd, self.tmpfn = tempfile.mkstemp(suffix='.pyout')

            if self.fd == STDOUT:
                self.old = sys.stdout
                sys.stdout = os.fdopen(self.tmpfd, 'w+b')
            else:
                self.old = sys.stderr
                sys.stderr = os.fdopen(self.tmpfd, 'w+b')

            self.started = True

    def flush(self):
        if self.fd == STDOUT:
            sys.stdout.flush()
        elif self.fd == STDERR:
            sys.stderr.flush()

    def stop(self):
        if self.started:
            self.flush()
            if self.fd == STDOUT:
                sys.stdout.close()
                sys.stdout = self.old
            else:
                sys.stderr.close()
                sys.stderr = self.old
            tmpr = open(self.tmpfn, 'rb')
            output = tmpr.read()
            tmpr.close()  # this also closes self.tmpfd
            os.remove(self.tmpfn)
            self.started = False
            return output
        else:
            return None
