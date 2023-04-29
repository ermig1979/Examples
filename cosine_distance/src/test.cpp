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

	void Test(const DescriptorFloat32Ptrs & original)
	{
		Buffer32f dist32f = MutualCosineDistances(original);
		Buffer32f dist16f = MutualCosineDistances(Convert<DescriptorFloat16>(original));
	}

	void CalcHist(const DescriptorFloat32 & desc, Buffer32u & hist, int lo, int hi, bool clear = true)
	{
		hist.resize(hi - lo + 1);
		if (clear)
		{
			for (size_t i = 0; i < hist.size(); ++i)
				hist[i] = 0;
		}
		for (size_t i = 0; i < desc.Size(); ++i)
		{
			int index = std::min(std::max((int)(desc.Data()[i] - lo), 0), hi - lo);
			hist[index]++;
		}
	}

	void PrintHist(const DescriptorFloat32Ptrs& desc, int lo, int hi)
	{
		std::cout << "Descriptor histograms:" << std::endl;
		Buffer32u hist;
		for (int i = lo; i <= hi; ++i)
			std::cout << ExpandLeft(std::to_string(i), 4);
		std::cout << std::endl;
		for (int i = lo; i <= hi; ++i)
			std::cout << "----";
		std::cout << std::endl;
		for (size_t i = 0; i < desc.size(); ++i)
		{
			CalcHist(*desc[i], hist, lo, hi, true);
			for (int j = lo; j <= hi; ++j)
				std::cout << ExpandLeft(std::to_string(hist[j - lo]), 4);
			std::cout << std::endl;
		}
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
	else if (mode == "file")
	{
		if (argc < 3)
			return 1;
		std::string path = argv[2];
		std::ifstream ifs(path.c_str());
		if (!ifs.is_open())
			return 1;
		cs::DescriptorFloat32Ptr desc = std::make_shared<cs::DescriptorFloat32>();
		while (desc->Load(ifs))
		{
			original.push_back(desc);
			desc = std::make_shared<cs::DescriptorFloat32>();
		}
		ifs.close();
	}
	else
		return 1;

	cs::PrintHist(original, -21, 21);
	cs::Test(original);

	return 0;
}