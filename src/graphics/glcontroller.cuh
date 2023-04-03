#pragma once

#include "model.cuh"

namespace graphics
{
	class GLController {
	public:

		GLController();
		void draw();

	private:
		Model particleModel;
	};
}