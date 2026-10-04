`timescale 1ns / 1ps

module tb_adder ();

    reg [7:0] a, b;
    wire [7:0] s;
    wire c;
    integer i, j;
    integer err, total;

    adder DUT (
        .a  (a),
        .b  (b),
        .s  (s),
        .c  (c)
    );

    initial begin
        err = 0;
        total = 0;
        // 입력 Driven 하는데 for문 쓰는 이유: 256가지 전체 테스트 시나리오를 보려면 일일히 다 입력하긴 힘듬.          
        for (i = 0; i < 256; i = i + 1) begin
            for (j = 0; j < 256; j = j + 1) begin
                a = i;
                b = j;
                #10;
                total = total + 1;
                if ({c, s} !== i + j) begin
                   err = err + 1;
                   $display("FAIL: %0d + %0d -> c=%b s=%0d (got %0d, expect %0d)", i,j,c,s, {c, s}, i + j); 
                end
            end
        end
        $display("TOTAL %0d, MISMATCH = %0d", total, err);
        $finish;
    end

endmodule
