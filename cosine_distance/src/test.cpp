#include "descriptor.h"

namespace cs
{
	template<class T> std::vector<std::shared_ptr<T>> Convert(const DescriptorFloat32Ptrs& src)
	{
		std::vector<std::shared_ptr<T>> dst(src.size());
		for (size_t i = 0; i < src.size(); ++i)
			dst[i] = std::make_shared<T>(*src[i]);
		return dst;
	}

	template<class T> Buffer32f MutualCosineDistances(const std::vector<std::shared_ptr<T>> & descriptors)
	{
		size_t count = descriptors.size();
		Buffer32f distances(Square(count));
		for (size_t i = 0; i < count; ++i)
		{
			for (size_t j = 0; j < count; ++j)
				distances[i * count + j] = descriptors[i]->CosineDistance(*descriptors[j]);
		}
		return distances;
	}

	void Test(const cs::DescriptorFloat32Ptrs & original)
	{
		Buffer32f dist32f = MutualCosineDistances(original);
		Buffer32f dist16f = MutualCosineDistances(Convert<DescriptorFloat16>(original));
	}
}

int main(int argc, char* argv[])
{
	if (argc < 2)
		return 1;
	std::string mode = argv[1];

	cs::DescriptorFloat32Ptrs original;
	if (mode == "random")
	{
		if (argc < 6)
			return 1;
		size_t count = std::stoi(argv[2]);
		size_t size = std::stoi(argv[3]);
		float lo = std::stof(argv[4]);
		float hi = std::stof(argv[5]);

		original.resize(count);
		for (size_t i = 0; i < count; ++i)
		{
			std::shared_ptr<cs::DescriptorFloat32> descriptor = std::make_shared<cs::DescriptorFloat32>();
			descriptor->Init(size, lo, hi);
			original[i] = descriptor;
		}
	}
	else
		return 1;

	cs::Test(original);

	return 0;
}