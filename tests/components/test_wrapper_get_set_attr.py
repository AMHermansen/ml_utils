import pytest
import torch

from ml_utils.components import BaseComponent, SwiGLUMLP, Wrapper


class MockWrapper(Wrapper):
    """A mock implementation of the Wrapper for testing purposes."""

    def forward(self, x, *args, **kwargs):
        return self.wrapped_component.forward(x, *args, **kwargs)


class Mock(BaseComponent):
    """A mock component to simulate behavior for testing."""

    def __init__(self):
        self.some_attribute = "default_value"

    @property
    def in_features(self) -> int | None:
        return 16

    @property
    def out_features(self) -> int:
        return 16

    def compute_something(self, x, extra_param=None):
        return torch.tensor([x]) + (torch.tensor([1]) if extra_param else torch.tensor([0]))

    def forward(self, x, *args, **kwargs):
        del args, kwargs  # Unused
        return x


class TestWrapper:
    """Test suite for the Wrapper class functionality."""

    def test_wrapper_initialization(self):
        """Test that wrapper initializes correctly with wrapped component."""
        # Create a mock component
        mock_component = SwiGLUMLP(10)

        # Create wrapper
        wrapper = MockWrapper(mock_component)

        # Test that attributes are properly set
        assert wrapper.wrapped_component == mock_component
        assert wrapper.in_features == 10
        assert wrapper.out_features == 10

    def test_attribute_delegation_getattr(self):
        """Test that attribute access is properly delegated to wrapped component."""
        mock_component = Mock()
        mock_component.some_attribute = "test_value"

        wrapper = MockWrapper(mock_component)

        # Test attribute access delegation
        assert wrapper.some_attribute == "test_value"

    def test_attribute_delegation_setattr(self):
        """Test that attribute setting is properly delegated to wrapped component."""
        mock_component = Mock()
        wrapper = MockWrapper(mock_component)

        # Set attribute on wrapper (should go to wrapped component)
        wrapper.new_attribute = "new_value"

        # Verify it was set on the wrapped component, not the wrapper
        assert hasattr(mock_component, 'new_attribute')
        assert mock_component.new_attribute == "new_value"
        assert wrapper.new_attribute is mock_component.new_attribute

    def test_local_attributes_preserved(self):
        """Test that local wrapper attributes are not delegated."""
        class CustomWrapper(Wrapper):
            _local_attrs = Wrapper._base_local_attrs | {'some_value'}

            def forward(self, x, *args, **kwargs):
                return self.wrapped_component.forward(x, *args, **kwargs)

        mock_component = Mock()
        mock_component.some_value = "original_value"
        wrapper = CustomWrapper(mock_component)

        # Set local attribute
        wrapper.some_value = "should_not_change"

        # Verify it was set locally, not on the original wrapped component
        assert wrapper.some_value == "should_not_change"
        assert mock_component.some_value != "should_not_change"

    def test_method_delegation(self):
        """Test that method calls are properly delegated."""
        mock_component = Mock()
        wrapper = MockWrapper(mock_component)
        result = wrapper.compute_something(5, extra_param=True)

    def test_nonexistent_attribute_handling(self):
        """Test that AttributeError is raised for nonexistent attributes."""
        mock_component = Mock()
        wrapper = MockWrapper(mock_component)

        # Configure mock to raise AttributeError for nonexistent attributes
        with pytest.raises(AttributeError):
            _ = mock_component.nonexistent_attr

        with pytest.raises(AttributeError):
            _ = wrapper.nonexistent_attr

    def test_forward_abstract(self):
        """Test that forward method is abstract and must be implemented."""
        mock_component = Mock()

        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            wrapper = Wrapper(mock_component)

    def test_complex_attribute_interaction(self):
        """Test complex interaction between wrapper and wrapped component attributes."""
        class TestComponent(BaseComponent):
            def __init__(self):
                self.weight = torch.randn(10, 10)
                self.bias = torch.randn(10)
                self.config = {"lr": 0.01, "momentum": 0.9}

            @property
            def in_features(self) -> int | None:
                return 10

            @property
            def out_features(self) -> int | None:
                return 10

            def get_parameters(self):
                return list(self.parameters())

            def parameters(self):
                return [self.weight, self.bias]

        component = TestComponent()

        # Create a concrete implementation for testing
        class ConcreteWrapper(Wrapper):
            def forward(self, x, *args, **kwargs):
                return self.wrapped_component.weight @ x + self.wrapped_component.bias

        wrapper = ConcreteWrapper(component)

        # Test attribute access
        assert torch.equal(wrapper.weight, component.weight)
        assert torch.equal(wrapper.bias, component.bias)
        assert wrapper.config == component.config

        # Test method delegation
        params = wrapper.get_parameters()
        assert len(params) == 2
        assert torch.equal(params[0], component.weight)
        assert torch.equal(params[1], component.bias)

        # Test attribute modification through wrapper
        new_weight = torch.randn(10, 10)
        wrapper.weight = new_weight
        assert torch.equal(component.weight, new_weight)

    def test_multiple_wrappers(self):
        """Test that multiple wrappers work correctly together."""
        mock_component = Mock()
        mock_component.value = "original"

        wrapper1 = MockWrapper(mock_component)
        wrapper2 = MockWrapper(wrapper1)

        # Test chained delegation
        assert wrapper2.value == "original"
        assert wrapper1.value == "original"

        # Test setting through chain
        wrapper2.value = "modified"
        assert mock_component.value == "modified"

    def test_local_attrs_class_variable(self):
        """Test that _local_attrs class variable works correctly."""
        # Test that subclasses can extend local_attrs
        class ExtendedWrapper(Wrapper):
            _local_attrs = Wrapper._base_local_attrs | {'extended_attr'}

            def forward(self, x, *args, **kwargs):
                return self.wrapped_component.forward(x, *args, **kwargs)

        mock_component = Mock()
        extended_wrapper = ExtendedWrapper(mock_component)

        # Set extended local attribute
        extended_wrapper.extended_attr = "local_value"

        # Verify it was set locally, not delegated
        assert extended_wrapper.extended_attr == "local_value"
        assert not hasattr(mock_component, 'extended_attr')

        # Test that normal attributes are still delegated
        extended_wrapper.delegated_attr = "delegated_value"
        assert mock_component.delegated_attr == "delegated_value"
