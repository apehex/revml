import tensorflow as tf

# OPCODES #####################################################################

STOP = 0x00
EQ = 0x14
EXTCODECOPY = 0x3C
BLOCKHASH = 0x40
COINBASE = 0x41
PREVRANDAO = 0x44
JUMPDEST = 0x5B
PUSH0 = 0x5F
PUSH32 = 0x7F
CREATE = 0xF0
CALLCODE = 0xF2
RETURN = 0xF3
DELEGATECALL = 0xF4
CREATE2 = 0xF5
REVERT = 0xFD
INVALID = 0xFE
SELFDESTRUCT = 0xFF

HALTING = [STOP, RETURN, REVERT, INVALID, SELFDESTRUCT]

is_halting = lambda opcode: opcode in HALTING
is_push = lambda opcode: opcode >= PUSH0 and opcode <= PUSH32

# INSTRUCTIONS ################################################################

def data_length(opcode: int) -> int:
    return is_push(opcode) * (opcode - PUSH0) # 0 if the opcode is not a push

def instruction_length(opcode: int) -> int:
    return 1 + data_length(opcode) # 1 byte for the opcode + n bytes of data

def iterate_over_instructions(bytecode: bytes) -> iter:
    __i = 0
    while __i < len(bytecode):
        __len = instruction_length(opcode=bytecode[__i])
        yield bytecode[__i:__i+__len]
        __i = __i + __len
