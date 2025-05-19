`timescale 1ns / 1ps

module MIPS_Processor_TB();
    // Testbench signals
    reg clk;
    reg reset;
    
    
    
    // Output register values for debugging
    wire [31:0] RegValue0, RegValue1, RegValue2, RegValue3;
    wire [31:0] RegValue4, RegValue5, RegValue6, RegValue7;
    wire [31:0] RegValue8, RegValue9, RegValue10, RegValue11;
    wire [31:0] RegValue12, RegValue13, RegValue14, RegValue15;
    wire [31:0] RegValue16, RegValue17, RegValue18, RegValue19;
    wire [31:0] RegValue20, RegValue21, RegValue22, RegValue23;
    wire [31:0] RegValue24, RegValue25, RegValue26, RegValue27;
    wire [31:0] RegValue28, RegValue29, RegValue30, RegValue31;
    
    // Instantiate the top-level MIPS processor
    MIPS_Processor uut (
        .clk(clk),
        .reset(reset),

        .RegValue0(RegValue0), .RegValue1(RegValue1), .RegValue2(RegValue2), .RegValue3(RegValue3),
        .RegValue4(RegValue4), .RegValue5(RegValue5), .RegValue6(RegValue6), .RegValue7(RegValue7),
        .RegValue8(RegValue8), .RegValue9(RegValue9), .RegValue10(RegValue10), .RegValue11(RegValue11),
        .RegValue12(RegValue12), .RegValue13(RegValue13), .RegValue14(RegValue14), .RegValue15(RegValue15),
        .RegValue16(RegValue16), .RegValue17(RegValue17), .RegValue18(RegValue18), .RegValue19(RegValue19),
        .RegValue20(RegValue20), .RegValue21(RegValue21), .RegValue22(RegValue22), .RegValue23(RegValue23),
        .RegValue24(RegValue24), .RegValue25(RegValue25), .RegValue26(RegValue26), .RegValue27(RegValue27),
        .RegValue28(RegValue28), .RegValue29(RegValue29), .RegValue30(RegValue30), .RegValue31(RegValue31)
    );
    
    // Access internal signals through hierarchical references
    wire [31:0] PC = uut.datapath.PC;
    wire [31:0] Instruction = uut.datapath.Instruction;
    wire [3:0] ALUControl = uut.datapath.ALUControl;
    wire Zero = uut.datapath.Zero;
    wire [31:0] ALUResult = uut.datapath.ALUResult;
    wire [31:0] ReadData = uut.datapath.ReadData;
    wire RegDst = uut.control.RegDst;
    wire Jump = uut.control.Jump;
    wire Branch = uut.control.Branch;
    wire MemRead = uut.control.MemRead;
    wire MemtoReg = uut.control.MemtoReg;
    wire [1:0] ALUOp = uut.control.ALUOp;
    wire MemWrite = uut.control.MemWrite;
    wire ALUSrc = uut.control.ALUSrc;
    wire RegWrite = uut.datapath.RegWrite;
    wire [4:0] WriteReg = uut.datapath.WriteReg;
    wire [31:0] WriteData = uut.datapath.WriteData;
    
    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns period (100MHz)
    end
    
    // Test sequence
    initial begin
        // Initialize signals


        
        // Apply reset
        #20;
        reset = 0;
        
     
        
      
        
        // Run simulation for enough cycles to complete 10 Fibonacci iterations
        // Each iteration takes approximately 9 instructions
        // We'll run for enough cycles to complete all iterations plus some extra
        #1500;
        
        // Display final state and Fibonacci results
        $display("Simulation Complete - Fibonacci Sequence Calculation");
        $display("------------------------------------------------");
        $finish;
    end

    // Monitor PC and register values during simulation
    always @(posedge clk) begin
        if (!reset) begin
            $display("Time: %t", $time);
            $display("PC: %h", PC);
            $display("Instruction: %h", Instruction);
            
            // Display control signals
            $display("Control Signals - RegDst: %b, Jump: %b, Branch: %b, MemRead: %b, MemtoReg: %b", 
                RegDst, Jump, Branch, MemRead, MemtoReg);
            $display("ALUOp: %b, MemWrite: %b, ALUSrc: %b, RegWrite: %b", 
                ALUOp, MemWrite, ALUSrc, RegWrite);
            
            // Display ALU operation and result
            $display("ALU Control: %b, Zero: %b, Result: %h", 
                ALUControl, Zero, ALUResult);
            
            // Display register values (only showing the ones used in this program)
            $display("Registers - $1: %d, $2: %d, $3: %d, $4: %d, $5: %d, $6: %d, $7: %d",
                RegValue1, // Constant 1
                RegValue2, // n
                RegValue3, // first
                RegValue4, // second/current fibonacci
                RegValue5, // next
                RegValue6, // counter
                RegValue7  // temp for branch condition
            );
            
            // If we're executing the store instruction at memory address 5,
            // display the Fibonacci number being stored
            if (PC == 32'd20) begin // PC = 5*4 = 20
                $display("Storing Fibonacci Number: %d at Data Memory Address 400", RegValue4);
            end
            
            $display("------------------------------------------------");
        end
    end
endmodule