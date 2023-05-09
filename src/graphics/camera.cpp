#include "camera.hpp"

#include <glm/ext/matrix_transform.hpp>

graphics::Camera::Camera()
{
	calculateView();
}

void graphics::Camera::moveLeft()
{
	position -= right * cameraMovementSpeed;
}

void graphics::Camera::moveRight()
{
	position += right * cameraMovementSpeed;
}

void graphics::Camera::moveForward()
{
	position += front * cameraMovementSpeed;
}

void graphics::Camera::moveBack()
{
	position -= front * cameraMovementSpeed;
}

void graphics::Camera::ascend()
{
	position += up * cameraMovementSpeed;
}

void graphics::Camera::descend()
{
	position -= up * cameraMovementSpeed;
}

void graphics::Camera::rotateLeft()
{}

void graphics::Camera::rotateRight()
{}

void graphics::Camera::rotateUp()
{}

void graphics::Camera::rotatateDown()
{}

glm::mat4 graphics::Camera::getView() const
{
	return view;
}

glm::vec3 graphics::Camera::getPosition() const
{
	return position;
}

void graphics::Camera::calculateView()
{
	view = glm::lookAt(position, position + front, up);
}
