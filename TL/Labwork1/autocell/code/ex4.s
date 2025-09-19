@ Type exercise 4 here

    INVOKE 1, 0, 1      

    SETI R2, #1        
    SETI R6, #0         

    SETI R3, #0        
L_X_LOOP:
    GOTO_GE L_END_X, R3, R0 

    SETI R4, #0        
L_Y_LOOP:
    GOTO_GE L_END_Y, R4, R1

    INVOKE 3, 3, 4     

    INVOKE 5, 7, 3      

    GOTO_NE L_SKIP_SET, R7, R2

    INVOKE 4, 2, 0      
L_SKIP_SET:

    ADD R4, R4, R2     
    GOTO L_Y_LOOP
L_END_Y:

    ADD R3, R3, R2    
    GOTO L_X_LOOP
L_END_X:

    STOP