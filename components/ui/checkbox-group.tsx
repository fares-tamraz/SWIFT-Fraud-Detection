"use client";

import { CheckboxGroup as CheckboxGroupPrimitive } from "@base-ui/react/checkbox-group";
import type React from "react";

function cn(...inputs: (string | undefined | null | false)[]) {
  return inputs.filter(Boolean).join(" ");
}

function mergeCheckboxGroupClassName(
  base: string,
  className: CheckboxGroupPrimitive.Props["className"]
): CheckboxGroupPrimitive.Props["className"] {
  if (typeof className === "function") {
    return (state) => cn(base, className(state));
  }
  return cn(base, className);
}

export function CheckboxGroup({
  className,
  ...props
}: CheckboxGroupPrimitive.Props): React.ReactElement {
  return (
    <CheckboxGroupPrimitive
      className={mergeCheckboxGroupClassName(
        "flex flex-col items-start gap-3",
        className
      )}
      {...props}
    />
  );
}

export { CheckboxGroupPrimitive };
