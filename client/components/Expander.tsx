import React, { useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

interface ExpanderProps {
  title: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}

export const Expander: React.FC<ExpanderProps> = ({
  title,
  defaultOpen = false,
  children,
}) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="expander">
      <button
        type="button"
        className="expander-header"
        onClick={() => setOpen((prev) => !prev)}
      >
        <span>{title}</span>
        {open ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
      </button>
      {open && <div className="expander-body">{children}</div>}
    </div>
  );
};
