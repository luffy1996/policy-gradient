#include <iostream>
#include <string>
using namespace std;

template <typename mytype>
mytype myfunction(mytype a , mytype b)
{	
	mytype result = a+b;
	return result;
}
int main(int argc, char const *argv[])
{
	/* code */
	// return 0;

	double a = 12.0, b =13.5235,out;
	out  = myfunction<double> (a,b);
	cout << out <<endl;
	
	int c = 12, d =13,out_;
	out_ = myfunction<int> (c,d);
	cout << out_ << endl;

	string A= "a" , B ="b" ,out__;
	out__ = myfunction<string> (A,B);
	cout << out__ << endl;

	return 0;
}