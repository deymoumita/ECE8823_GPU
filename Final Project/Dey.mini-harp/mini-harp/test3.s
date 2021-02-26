/*
Divergent loop
*/

.perm x
.entry
.global
entry:
        ldi %r0, test;

        ldi %r1, #1;
        wspawn %r1, %r0, %r1;
        ldi %r1, #0;
        
        jmpi test;

.global
test:  
        
        ori %r5, %r1, #0;
        
        iszero @p0, %r5;
    @p0 ? ldi %r3, #2;
        subi %r2, %r5, #1;
        iszero @p0, %r2;
    @p0 ? ldi %r3, #5; 

        ldi %r2, #1;
        clone %r2;
        ldi %r2, #2;
        clone %r2;
        ldi %r2, #3;
        clone %r2;
        ldi %r2, #0;

        ldi %r6, #4;
        jalis %r7, %r6, diverge;

        subi %r1, %r5, #1;
        rtop @p0, %r1;
    @p0 ? halt; 

        ldi %r2, Output0;        
        ld %r0, %r2, (__WORD*0);                
        jali %r7, printhex;     
        jali %r7, printnl;
        ldi %r2, Output0;
        ld %r0, %r2, (__WORD*1);        
        jali %r7, printhex;     
        jali %r7, printnl;
        ldi %r2, Output0;
        ld %r0, %r2, (__WORD*2);        
        jali %r7, printhex;     
        jali %r7, printnl;
        ldi %r2, Output0;
        ld %r0, %r2, (__WORD*3);        
        jali %r7, printhex;     
        jali %r7, printnl; 

        jali %r7, printnl; 

        ldi %r2, Output1;        
        ld %r0, %r2, (__WORD*0);                
        jali %r7, printhex;     
        jali %r7, printnl;
        ldi %r2, Output1;
        ld %r0, %r2, (__WORD*1);        
        jali %r7, printhex;     
        jali %r7, printnl;
        ldi %r2, Output1;
        ld %r0, %r2, (__WORD*2);        
        jali %r7, printhex;     
        jali %r7, printnl;
        ldi %r2, Output1;
        ld %r0, %r2, (__WORD*3);        
        jali %r7, printhex;     
        jali %r7, printnl; 

        halt;
        
diverge: 
        tid %r0;
        ldi %r4, #0;
        addi %r1, %r0, #1;

    loop: add %r4, %r4, %r3;
        subi %r1, %r1, #1;          /* update loop counter */
        rtop @p0, %r1; 
        split;             
  @p0 ? jmpi loop; 
        join;


        iszero @p0, %r5;
    @p0 ? ldi %r0, Output0;
        subi %r3, %r5, #1;
        iszero @p0, %r3;
    @p0 ? ldi %r0, Output1;
        
        shli %r2, %r2, #2;
        add %r0, %r0, %r2;
        st %r4, %r0, #0;
        jmprt %r7;


.perm rw
Output0:  .word 0 0 0 0

.perm rw
Output1:  .word 0 0 0 0

