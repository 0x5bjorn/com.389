Flag = `pkg-config --cflags --libs opencv4 tesseract` -llept -ltesseract

all: tinyexpr MathExpressionSolverUsingCamera

MathExpressionSolverUsingCamera: MathExpressionSolverUsingCamera.cpp
	g++ -g -std=c++11 MathExpressionSolverUsingCamera.cpp tinyexpr.o -o MathExpressionSolverUsingCamera $(Flag)

tinyexpr: tinyexpr.c
	gcc -c tinyexpr.c

# MathExpressionSolverUsingCamera.o: MathExpressionSolverUsingCamera.cpp
# 	g++ -c MathExpressionSolverUsingCamera.cpp

# output: MathExpressionSolverUsingCamera.o
# 	g++ MathExpressionSolverUsingCamera.o -o MathExpressionSolverUsingCamera

clean:
	rm -rf MathExpressionSolverUsingCamera tinyexpr.o