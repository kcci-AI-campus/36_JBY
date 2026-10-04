`timescale 1ns / 1ps

module tb_adder4_fnd ();

    reg [3:0] a, b;
    wire [3:0] fnd_com;
    wire [7:0] fnd_font;
    wire led;

    adder4_fnd DUT (
        .a(a),
        .b(b),
        .fnd_com(fnd_com),
        .fnd_font(fnd_font),
        .led(led)
    );

    integer i, j;
    integer err;
    reg [4:0] expect;

    initial begin
        err = 0;
        for (i = 0; i < 16; i = i + 1) begin
            for (j = 0; j < 16; j = j + 1) begin
                a = i;
                b = j;
                #10;
                expect = i + j;
                if ({led, DUT.w_bin} !== expect) begin
                    err = err + 1;
                    $display("FAIL: %0d + %0d -> led=%b s=%0d (expect %0d)", i, j, led, DUT.w_bin, expect);
                end
            end
        end
        $display("TOTAL 256, MISMATCH = %0d", err);
        $finish;
    end

endmodule
