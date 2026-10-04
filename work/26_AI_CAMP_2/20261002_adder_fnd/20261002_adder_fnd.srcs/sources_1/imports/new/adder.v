`timescale 1ns / 1ps
//자동정렬: 컨트롤 쉬프트 P -> system verilog formatter - format this file
//`default_nettype none


module adder_fnd (
    input [7:0] a,
    input [7:0] b,
    input [1:0] ds_sel,
    output [3:0] fnd_com,
    output [7:0] fnd_font
);

    wire [7:0] w_s;
    wire w_c;

    adder U_ADDER (
    .a(a),
    .b(b),
    .s(w_s),
    .c(w_c)
);

fnd_controller U_FND_CNTL (
    .ds_sel(ds_sel),
    .fnd_data({w_c, w_s}),  // 9bit {c, s}
    .fnd_com(fnd_com),
    .fnd_font(fnd_font)
);

endmodule



module adder (
    input [7:0] a,
    input [7:0] b,
    input cin,
    output [7:0] s,
    output c
);

    wire w_c;

    full_adder_4 FA4_2 (
        .a  (a[7:4]),
        .b  (b[7:4]),
        .cin(w_c),
        .s  (s[7:4]),
        .c  (c)
    );

    full_adder_4 FA4_1 (
        .a  (a[3:0]),
        .b  (b[3:0]),
        .cin(1'b0),
        .s  (s[3:0]),
        .c  (w_c)
    );

endmodule



module full_adder_4 (
    input [3:0] a,
    input [3:0] b,
    input cin,
    output [3:0] s,
    output c
);
    //변수 선언시 항상 비트 규격 잡아라.
    wire w_c1, w_c2, w_c3; // wire 변수 선언 안해주면 1bit로 자동 선언되는 거 조심. 무조건 비트 폭 잘 생각해서 무조건 wire 변수 선언해라. 

    // instanciation, 실체화
    // 구조적 설계
    full_adder U_FA4 (
        .a  (a[3]),
        .b  (b[3]),
        .cin(w_c3),
        .s  (s[3]),
        .c  (c)
    );

    full_adder U_FA3 (
        .a  (a[2]),
        .b  (b[2]),
        .cin(w_c2),
        .s  (s[2]),
        .c  (w_c3)
    );

    full_adder U_FA2 (
        .a  (a[1]),
        .b  (b[1]),
        .cin(w_c1),
        .s  (s[1]),
        .c  (w_c2)
    );

    full_adder U_FA1 (
        .a  (a[0]),
        .b  (b[0]),
        .cin(cin),
        .s  (s[0]),
        .c  (w_c1)
    );
endmodule

module full_adder (
    input  a,
    input  b,
    input  cin,
    output s,
    output c
);

    wire w_s1; // reg가 아닌 이유: reg는 값을 drive 하는데, 안에서도 drive하고 밖에서도 drive하면 쇼트 남. 시뮬레이션에서 X가 나오면 쇼트·미연결·초기값 없음부터 의심 
    wire w_c1, w_c2;

    assign c = w_c1 | w_c2;

    half_adder U_HA1 (
        .a(a),
        .b(b),
        .s(w_s1),
        .c(w_c1)
    );

    half_adder U_HA2 (
        .a(w_s1),
        .b(cin),
        .s(s),  // full adder output s
        .c(w_c2)
    );


endmodule

module half_adder (
    input  a,
    input  b,
    output s,
    output c
);

    assign s = a ^ b;
    assign c = a & b;

endmodule
