/*
This program write otu 'Hello World' to the console output.
*/

.perm x
.entry
.global
entry:	ldi %r0, Hello;
        jali %r7, prints;
        halt;

.perm r
Hello:
.string "Hello World\n"
