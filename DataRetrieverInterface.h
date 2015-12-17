#ifndef DATA_RETRIEVER_INTERFACE_H
#define DATA_RETRIEVER_INTERFACE_H

template<typename T>
class DataRetrieverInterface
{
public:
	virtual bool getNextPlane(T **data, int offset) = 0;
};

#endif

