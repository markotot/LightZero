#!/usr/bin/expect

set APOC_USERNAME [lindex $argv 0]
set APOC_PASSPHRASE [lindex $argv 1];
set APOC_PASSWORD [lindex $argv 2];
set APOC_PRIVATE_KEY [lindex $argv 3];

ssh -i $APOC_PRIVATE_KEY $APOC_USERNAME@andrena.hpc.qmul.ac.uk
