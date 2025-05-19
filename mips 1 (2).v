`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 21.04.2025 13:52:32
// Design Name: 
// Module Name: mips
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

//// Basic gate-level components (keeping the same style as provided code)
module mux2x1_gate_level (a,b,sel,y);
    input a, b; // Data inputs
    input sel; // Selection input
    output y; // Output

    assign y = sel ? b:a; // Output from the OR gate
endmodule 

module mux2x1_5bit (a,b,sel,y);
    input[4:0] a, b; // Data inputs
    input sel; // Selection input
    output[4:0] y; // Output

    assign y = sel ? b:a; // Output from the OR gate
endmodule 
// 32-bit 2:1 MUX
module mux2x1_32bit (y, a, b, sel);
    input [31:0] a;
    input [31:0] b;
    input sel;
    output [31:0] y;

    // Gate-level implementation of the 32-bit multiplexer
    genvar i;
    generate
        for (i = 0; i < 32; i = i + 1) begin: mux_gen
            mux2x1_gate_level mux_inst(.a(a[i]), .b(b[i]), .sel(sel), .y(y[i]));
        end
    endgenerate
endmodule

// Half adder at gate level
module halfadder (S, C, x, y);
    input x, y;
    output S, C;
    xor (S, x, y);
    and (C, x, y);
endmodule

// Full adder at gate level
module fulladder (S, C, x, y, z);
    input x, y, z;
    output S, C;
    wire S1, D1, D2;
    halfadder HA1 (S1, D1, x, y);
    halfadder HA2 (S, D2, S1, z);
    or g1 (C, D2, D1);
endmodule

// 32-bit adder
module thirtytwo_bit_adder (S, C32, A, B, Cin);
    input [31:0] A, B;
    input Cin;
    output [31:0] S;
    output C32;
    wire [31:0] C;
    // First full adder with carry-in
    fulladder FA0 (S[0], C[0], A[0], B[0], Cin);
    
    // Generate the remaining full adders
    genvar i;
    generate
        for (i = 1; i < 32; i = i + 1) begin: adder_gen
            fulladder FA (S[i], C[i], A[i], B[i], C[i-1]);
        end
    endgenerate
    
    assign C32 = C[31]; // Final carry out
endmodule

// ALU module supporting add, sub, and, or, slt operations
module ALU_32bit(
    input [31:0] A,       // 32-bit input operand A
    input [31:0] B,       // 32-bit input operand B
    input [3:0] ALUControl,  // Control signal for operation selection
    output [31:0] Result,   // 32-bit output result
    output Zero            // 1 if Result is zero, 0 otherwise
);
    wire [31:0] B_modified;
    wire [31:0] add_result, and_result, or_result, slt_result;
    wire [31:0] not_B;
    wire cin;
    wire cout;
    wire set;
    
    // NOT operation on B for subtraction
    genvar i;
    generate
        for (i = 0; i < 32; i = i + 1) begin: not_gen
            not(not_B[i], B[i]);
        end
    endgenerate
    
    // Select B or ~B based on ALUControl
    assign cin = (ALUControl[2] == 1'b1) ? 1'b1 : 1'b0; // 1 for subtraction
    mux2x1_32bit mux_b_sel (.y(B_modified), .a(B), .b(not_B), .sel(ALUControl[2]));
    
    // Adder for add and subtract operations
    thirtytwo_bit_adder adder(.S(add_result), .C32(cout), .A(A), .B(B_modified), .Cin(cin));
    
    // AND operation
    genvar j;
    generate
        for (j = 0; j < 32; j = j + 1) begin: and_gen
            and(and_result[j], A[j], B[j]);
        end
    endgenerate
    
    // OR operation
    genvar k;
    generate
        for (k = 0; k < 32; k = k + 1) begin: or_gen
            or(or_result[k], A[k], B[k]);
        end
    endgenerate
    
    // SLT operation (Set if Less Than)
    // SLT sets result to 1 if A < B (signed comparison), else 0
    assign set = add_result[31]; // Sign bit of subtraction result
    assign slt_result = {31'b0, set};
    
    // MUX to select final result based on ALUControl
    wire [31:0] temp_result1, temp_result2;
    mux2x1_32bit mux_result1 (.y(temp_result1), .a(and_result), .b(or_result), .sel(ALUControl[0]));
    mux2x1_32bit mux_result2 (.y(temp_result2), .a(add_result), .b(slt_result), .sel(ALUControl[0]));
    mux2x1_32bit mux_result_final (.y(Result), .a(temp_result1), .b(temp_result2), .sel(ALUControl[1]));
    
    // Zero flag
    wire [31:0] or_stages[4:0];
    
    // First stage - OR gates for every 2 bits
    genvar m;
    generate
        for (m = 0; m < 16; m = m + 1) begin: zero_stage1
            or(or_stages[0][m], Result[2*m], Result[2*m + 1]);
        end
    endgenerate
    
    // Second stage - OR gates for every 2 results from first stage
    genvar n;
    generate
        for (n = 0; n < 8; n = n + 1) begin: zero_stage2
            or(or_stages[1][n], or_stages[0][2*n], or_stages[0][2*n + 1]);
        end
    endgenerate
    
    // Third stage
    genvar p;
    generate
        for (p = 0; p < 4; p = p + 1) begin: zero_stage3
            or(or_stages[2][p], or_stages[1][2*p], or_stages[1][2*p + 1]);
        end
    endgenerate
    
    // Fourth stage
    genvar q;
    generate
        for (q = 0; q < 2; q = q + 1) begin: zero_stage4
            or(or_stages[3][q], or_stages[2][2*q], or_stages[2][2*q + 1]);
        end
    endgenerate
    
    // Final stage
    or(or_stages[4][0], or_stages[3][0], or_stages[3][1]);
    
    // Invert to get Zero flag
    not(Zero, or_stages[4][0]);
endmodule

// Register File module (32 registers x 32 bits each)
module RegisterFile(
    input clk, reset,
    input RegWrite,
    input [4:0] ReadReg1, ReadReg2, WriteReg,
    input [31:0] WriteData,
    output [31:0] ReadData1, ReadData2
);
    reg [31:0] registers [31:0];
    
    initial begin  registers[0] <= 32'd0; registers[1]<=32'd1;end
   
    // Write operation
    always @(posedge clk) begin
        if (RegWrite)
            registers[WriteReg] <= WriteData;
    end
    // Read operations
    assign ReadData1 =  registers[ReadReg1];
    assign ReadData2 =  registers[ReadReg2];
endmodule

module InstructionMemory(
    input [31:0] PC,
    output [31:0] Instruction,
    // New interface for loading instructions from testbench
    input write_enable,
    input [7:0] write_addr,
    input [31:0] write_data
);
    reg [31:0] memory [255:0]; // 256 words of 32-bit memory
    
    // Address is word-aligned and limited to memory size
    wire [7:0] word_addr = PC[9:2];
    
    // Read operation (asynchronous)
    assign Instruction = memory[word_addr];
        initial begin
        // Load n from memory[0] 
        memory[0] = 32'b100011_00000_00010_0000000000000000; // lw $2, 0($0) ; n
        memory[1] = 32'b000000_00000_00000_00011_00000_100000; // add $3, $0, $0 ; first = 0 
        memory[2] = 32'b000000_00000_00001_00100_00000_100000; // add $4, $0, $1 ; second = 1 (use $1 as temp) 
        memory[3] = 32'b000000_00000_00000_00110_00000_100000; // add $6, $0, $0 ; counter = 0 // Label: loop
        memory[4] = 32'b101011_00001_00011_0000000000000010; // sw $3, 2($1) ; store first
          
        memory[5] = 32'b000000_00011_00100_00101_00000_100000; // add $5, $3, $4 ; next = first + second
        memory[6] = 32'b000000_00100_00000_00011_00000_100000; // add $3, $4, $0 ; first = second 
        memory[7] = 32'b000000_00101_00000_00100_00000_100000; // add $4, $5, $0 ; second = next
        memory[8] = 32'b000000_00110_00001_00110_00000_100000; // add $6, $6, $1 ; counter += 1 
        memory[9] = 32'b000000_00110_00000_00111_00000_100000; // add $7, $6, $0 ; temp = counter 
        memory[10] = 32'b000000_00111_00010_00111_00000_101010; // slt $7, $7, $2 ; if counter < n
        memory[11] = 32'b101011_00000_00100_0000000110010000; // sw $4, 400($0)
        memory[12] = 32'b000100_00111_00001_1111111111111000; // be $7, $1, -8 ; branch to loop (PC-relative)
        memory[13] = 32'b000100_00000_00000_1111111111111111; // beq $0, $0, loop (offset = -1) loop at done
        end
    
    
endmodule



module DataMemory(
    input clk,
    input MemWrite, MemRead,
    input [31:0] Address, WriteData,
    output [31:0] ReadData

);
    reg [31:0] memory [255:0]; // 256 words of 32-bit memory
    integer i;
    
    
    // Normal write operation during execution
    always @(posedge clk) begin
        if (MemWrite)
            memory[Address[9:2]] <= WriteData;
    end
    // hard wire some values
    initial begin
    memory[0] = 32'd10;
    end
    
    // Read operation
    assign ReadData = MemRead ? memory[Address[9:2]] : 32'b0;
endmodule

// Control Unit to decode instructions and generate control signals
module Control(
    input [5:0] Opcode,
    output RegDst, Jump, Branch, MemRead, MemtoReg, 
    output [1:0] ALUOp,
    output MemWrite, ALUSrc, RegWrite
);
    // Main decoder for the control signals
    reg [9:0] controls;
    
    always @(*) begin
        case(Opcode)
            6'b000000: controls = 10'b1001000100; // R-type
            6'b100011: controls = 10'b0111100000; // LW
            6'b101011: controls = 10'b0100010000; // SW
            6'b000100: controls = 10'b0000001010; // BEQ
            6'b001000: controls = 10'b0101000000; // ADDI
            6'b000010: controls = 10'b0000000001; // J
            6'b000011: controls = 10'b0001000001; // JAL
            default:   controls = 10'b0000000000; // Illegal op
        endcase
    end
    
    assign {RegDst, ALUSrc, MemtoReg, RegWrite, MemRead, MemWrite, Branch, ALUOp, Jump} = controls;
endmodule

// ALU Control Unit
module ALUControl(
    input [1:0] ALUOp,
    input [5:0] Funct,
    output [3:0] ALUControl
);
    reg [3:0] aluctl;
    
    always @(*) begin
        case(ALUOp)
            2'b00: aluctl = 4'b0010; // Add (for lw/sw/addi)
            2'b01: aluctl = 4'b0110; // Subtract (for beq)
            2'b10: begin // R-type
                case(Funct)
                    6'b100000: aluctl = 4'b0010; // ADD
                    6'b100010: aluctl = 4'b0110; // SUB
                    6'b100100: aluctl = 4'b0000; // AND
                    6'b100101: aluctl = 4'b0001; // OR
                    6'b101010: aluctl = 4'b0111; // SLT
                    default:   aluctl = 4'b0000;
                endcase
            end
            default: aluctl = 4'b0000;
        endcase
    end
    
    assign ALUControl = aluctl;
endmodule

// Sign Extension module
module SignExtend(
    input [15:0] in,
    output [31:0] out
);
    assign out = {{16{in[15]}}, in};
endmodule

// PC Register
module ProgramCounter(
    input clk, reset,
    input [31:0] PC_next,
    output reg [31:0] PC
);
    // Initialize PC
    initial begin
        PC = 32'b0;
    end
    
    always @(posedge clk) begin
        if (reset)
            PC <= 32'b0;
        else
            PC <= PC_next;
    end
endmodule

// Datapath Unit
module Datapath(
    input clk, reset,
    // Control signals
    input RegDst, Jump, Branch, MemRead, MemtoReg, 
    input [1:0] ALUOp,
    input MemWrite, ALUSrc, RegWrite,
    // Outputs to Control
    output [5:0] Opcode,
    output Zero,
    // New testbench interfaces for memory initialization
    // Instruction Memory initialization interface
    
    output [31:0] RegValue0, RegValue1, RegValue2, RegValue3,
        output [31:0] RegValue4, RegValue5, RegValue6, RegValue7,
        output [31:0] RegValue8, RegValue9, RegValue10, RegValue11,
        output [31:0] RegValue12, RegValue13, RegValue14, RegValue15,
        output [31:0] RegValue16, RegValue17, RegValue18, RegValue19,
        output [31:0] RegValue20, RegValue21, RegValue22, RegValue23,
        output [31:0] RegValue24, RegValue25, RegValue26, RegValue27,
        output [31:0] RegValue28, RegValue29, RegValue30, RegValue31
        

);
    // Internal wires
    wire [31:0] PC, PC_next, PC_plus4, PC_branch, PC_jump;
    wire [31:0] Instruction;
    wire [4:0] WriteReg;
    wire [31:0] WriteData, ReadData1, ReadData2;
    wire [31:0] SignImm, ShiftedImm;
    wire [31:0] SrcB, ALUResult;
    wire [31:0] ReadData;
    wire [3:0] ALUControl;
    wire PCSrc;
    wire check;
    
    // Extract operation code from instruction
    assign Opcode = Instruction[31:26];
    
    // PC Logic
    ProgramCounter pc_reg(.clk(clk), .reset(reset), .PC_next(PC_next), .PC(PC));
    thirtytwo_bit_adder pc_adder(.S(PC_plus4), .C32(check), .A(PC), .B(32'd4), .Cin(1'b0));
    wire [31:0] nextImm;
    wire check2;
    // PC source multiplexer
    assign PCSrc = Branch & Zero;
    mux2x1_32bit pc_src_mux(.y(PC_branch), .a(PC_plus4), .b(nextImm), .sel(PCSrc));
    thirtytwo_bit_adder sec(.S(nextImm), .C32(check2), .A(PC_plus4), .B(ShiftedImm), .Cin(1'b0));
    // Jump address calculation
    assign PC_jump = {PC_plus4[31:28], Instruction[25:0], 2'b00};
    mux2x1_32bit jump_mux(.y(PC_next), .a(PC_branch), .b(PC_jump), .sel(Jump));
    
    
    InstructionMemory imem(
        .PC(PC), 
        .Instruction(Instruction)
    );
    
    // Register file logic
    mux2x1_5bit reg_dst_mux(.a(Instruction[20:16]), .b(Instruction[15:11]), .sel(RegDst), .y(WriteReg));
    RegisterFile regfile(
        .clk(clk), .reset(reset),
        .RegWrite(RegWrite),
        .ReadReg1(Instruction[25:21]), .ReadReg2(Instruction[20:16]), 
        .WriteReg(WriteReg),
        .WriteData(WriteData),
        .ReadData1(ReadData1), .ReadData2(ReadData2)
    );
    
    assign RegValue0 = regfile.registers[0];
    assign RegValue1 = regfile.registers[1];
    assign RegValue2 = regfile.registers[2];
    assign RegValue3 = regfile.registers[3];
    assign RegValue4 = regfile.registers[4];
    assign RegValue5 = regfile.registers[5];
    assign RegValue6 = regfile.registers[6];
    assign RegValue7 = regfile.registers[7];
    assign RegValue8 = regfile.registers[8];
    assign RegValue9 = regfile.registers[9];
    assign RegValue10 = regfile.registers[10];
    assign RegValue11 = regfile.registers[11];
    assign RegValue12 = regfile.registers[12];
    assign RegValue13 = regfile.registers[13];
    assign RegValue14 = regfile.registers[14];
    assign RegValue15 = regfile.registers[15];
    assign RegValue16 = regfile.registers[16];
    assign RegValue17 = regfile.registers[17];
    assign RegValue18 = regfile.registers[18];
    assign RegValue19 = regfile.registers[19];
    assign RegValue20 = regfile.registers[20];
    assign RegValue21 = regfile.registers[21];
    assign RegValue22 = regfile.registers[22];
    assign RegValue23 = regfile.registers[23];
    assign RegValue24 = regfile.registers[24];
    assign RegValue25 = regfile.registers[25];
    assign RegValue26 = regfile.registers[26];
    assign RegValue27 = regfile.registers[27];
    assign RegValue28 = regfile.registers[28];
    assign RegValue29 = regfile.registers[29];
    assign RegValue30 = regfile.registers[30];
    assign RegValue31 = regfile.registers[31];
    
    // Sign extension and shift
    SignExtend signext(.in(Instruction[15:0]), .out(SignImm));
    
    // Shift left 2 for branch target
    assign ShiftedImm = {SignImm[29:0], 2'b00};
    
    // ALU logic
    ALUControl alucontrol(.ALUOp(ALUOp), .Funct(Instruction[5:0]), .ALUControl(ALUControl));
    mux2x1_32bit alu_src_mux(.y(SrcB), .a(ReadData2), .b(SignImm), .sel(ALUSrc));
    ALU_32bit alu(.A(ReadData1), .B(SrcB), .ALUControl(ALUControl), .Result(ALUResult), .Zero(Zero));
    
    // Data Memory Initialisation
    DataMemory dmem(
        .clk(clk),
        .MemWrite(MemWrite), .MemRead(MemRead),
        .Address(ALUResult), .WriteData(ReadData2),
        .ReadData(ReadData)

    );
    
    // Write back to register file
    mux2x1_32bit mem_to_reg_mux(.y(WriteData), .a(ALUResult), .b(ReadData), .sel(MemtoReg));
endmodule

// Top-level module for MIPS processor
module MIPS_Processor(
    input clk, reset,
    


    // Register values for debugging
    output [31:0] RegValue0, RegValue1, RegValue2, RegValue3,
    output [31:0] RegValue4, RegValue5, RegValue6, RegValue7,
    output [31:0] RegValue8, RegValue9, RegValue10, RegValue11,
    output [31:0] RegValue12, RegValue13, RegValue14, RegValue15,
    output [31:0] RegValue16, RegValue17, RegValue18, RegValue19,
    output [31:0] RegValue20, RegValue21, RegValue22, RegValue23,
    output [31:0] RegValue24, RegValue25, RegValue26, RegValue27,
    output [31:0] RegValue28, RegValue29, RegValue30, RegValue31
);
    // Internal wires
    wire [5:0] Opcode;
    wire Zero;
    
    // Control signals
    wire RegDst, Jump, Branch, MemRead, MemtoReg;
    wire [1:0] ALUOp;
    wire MemWrite, ALUSrc, RegWrite;
    
    // Control Unit
    Control control(
        .Opcode(Opcode),
        .RegDst(RegDst), .Jump(Jump), .Branch(Branch),
        .MemRead(MemRead), .MemtoReg(MemtoReg),
        .ALUOp(ALUOp),
        .MemWrite(MemWrite), .ALUSrc(ALUSrc), .RegWrite(RegWrite)
    );
    
    // Datapath
    Datapath datapath(
        .clk(clk), .reset(reset),
        .RegDst(RegDst), .Jump(Jump), .Branch(Branch),
        .MemRead(MemRead), .MemtoReg(MemtoReg),
        .ALUOp(ALUOp),
        .MemWrite(MemWrite), .ALUSrc(ALUSrc), .RegWrite(RegWrite),
        .Opcode(Opcode),
        .Zero(Zero),
        .RegValue0(RegValue0), .RegValue1(RegValue1), .RegValue2(RegValue2), .RegValue3(RegValue3),
        .RegValue4(RegValue4), .RegValue5(RegValue5), .RegValue6(RegValue6), .RegValue7(RegValue7),
        .RegValue8(RegValue8), .RegValue9(RegValue9), .RegValue10(RegValue10), .RegValue11(RegValue11),
        .RegValue12(RegValue12), .RegValue13(RegValue13), .RegValue14(RegValue14), .RegValue15(RegValue15),
        .RegValue16(RegValue16), .RegValue17(RegValue17), .RegValue18(RegValue18), .RegValue19(RegValue19),
        .RegValue20(RegValue20), .RegValue21(RegValue21), .RegValue22(RegValue22), .RegValue23(RegValue23),
        .RegValue24(RegValue24), .RegValue25(RegValue25), .RegValue26(RegValue26), .RegValue27(RegValue27),
        .RegValue28(RegValue28), .RegValue29(RegValue29), .RegValue30(RegValue30), .RegValue31(RegValue31)


    );
endmodule
