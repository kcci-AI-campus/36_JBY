`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2026/09/30 10:43:33
// Design Name: 
// Module Name: gates
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


module gates (
    input  a,       // input wire a , wire 생략 됨. 1bit임. 기본 단위 bit(하드웨어라서). sw c언어 기본단위 바이트랑 다름. 하드웨어는 물리적인 와이어를 차지하기 때문에, 공간은 곧 비용이고 칩 사이즈가 증가하기 때문에 작은 비트단위로 제어.
    input  b,
    output y0,
    output y1,
    output y2,
    output y3,
    output y4,
    output y5,
    output y6
);

    assign y0 = a & b;  // a and b
    assign y1 = ~(a & b);  // a nand b
    assign y2 = a | b;  // a or b
    assign y3 = ~(a | b);  // a nor b
    assign y4 = a ^ b;  // a exor b
    assign y5 = ~(a ^ b);  // a exnor b
    assign y6 = ~a;  // not a

// assign: 한번에 이뤄짐.실행 순서가 없음.  이유: hw 구성으로 배치하는 거임. 연결해서 로직 구성하는 걸 알려주는 거임. 


endmodule
