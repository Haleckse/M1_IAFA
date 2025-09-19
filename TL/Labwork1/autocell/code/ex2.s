@ Type exercise 2 here

	INVOKE 1, 0, 1 
    SETI R2, #1
    SETI R5, #1

    SETI R3, #0
    SETI R4, #0
    INVOKE 3, 3, 4
    INVOKE 4, 2, 0

    SUB R3, R0, R5
    SETI R4, #0
    INVOKE 3, 3, 4
    INVOKE 4, 2, 0

    SETI R3, #0
    SUB R4, R1, R5
    INVOKE 3, 3, 4
    INVOKE 4, 2, 0

    SUB R3, R0, R5
    SUB R4, R1, R5
    INVOKE 3, 3, 4
    INVOKE 4, 2, 0

    STOP
