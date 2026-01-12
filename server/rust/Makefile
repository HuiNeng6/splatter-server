CARGO ?= cargo
PACKAGE := splatter-bin

.PHONY: run check fmt clippy test ci

run:
	$(CARGO) run -p $(PACKAGE)

check:
	$(CARGO) check -p $(PACKAGE)

fmt:
	$(CARGO) fmt --all

clippy:
	$(CARGO) clippy -p $(PACKAGE) --all-targets -- -D warnings

test:
	$(CARGO) test -p $(PACKAGE)

ci: fmt clippy test
	@echo "CI checks passed."
