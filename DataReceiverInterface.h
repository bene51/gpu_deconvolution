#ifndef DATA_RECEIVER_INTERFACE_H
#define DATA_RECEIVER_INTERFACE_H

template<typename T>
class DataReceiverInterface
{
public:
	virtual void returnNextPlane(T *data) = 0;
};
#endif


