`timescale 1ns / 1ps

module tb_gates();
//reg, wire: 1bit 자료형
    reg a , b; // 변수. 내부 모듈 입력을 드라이브할. 값을 Drive 가능, 값을 유지할 수 있음.(스스로) 
    wire y0,y1,y2,y3,y4,y5,y6; // 메탈 라인. 연결할 때만 씀. 
    //Multiple Driven Error: 값을 이쪽에서는 1을 주고 저쪽에서는 -1을 주고 그러면 쇼트가 날 수 있다. 그래서 출력을 이쪽 저쪽에서 드라이브 하려고 하면 안된다. 그래서 reg가 아니라 wire 선언한다.
    //입력은 reg로 드라이브 하고  , 출력은 wire로 연결만한다,.
    gates DUT( // DUT: Design Under Test = 인스턴스(실체화) 이름
        .a(a),       
        .b(b),
        .y0(y0),
        .y1(y1),
        .y2(y2),
        .y3(y3),
        .y4(y4),
        .y5(y5),
        .y6(y6)
    );

    initial begin
        // 1. 시간 Delay
        // 2. 값을 Drive(변경)
        a = 0; 
        b = 0;

        #1; // delay 1ns 1ns로 가려면 1ps가 1000번 가야됨. 즉, 1000번 간 거임. 그 사이에 뭔 일이 발생할지 몰라서 해석 단위를 1ns가 아닌 1ps

        a = 1;
        b = 0;

        #1;

        a = 0;
        b = 1;

        #1;
        a = 1;
        b = 1;
        // 뒤에 딜레이 안주면 a=1, b=1에 대한 결과는 못봄. 결과를 보려면 시간을 뒤에서도 줘야 함. 
        #1;

        $stop; // 시뮬레이션 중지

    end

endmodule
