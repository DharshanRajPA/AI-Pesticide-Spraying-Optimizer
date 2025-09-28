import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import styled from 'styled-components';
import { 
  Home, 
  BarChart3, 
  FileText, 
  Settings, 
  LogOut,
  Shield,
  Zap
} from 'lucide-react';

const HeaderContainer = styled.header`
  background: ${props => props.theme.colors.surface};
  border-bottom: 1px solid ${props => props.theme.colors.border};
  box-shadow: ${props => props.theme.shadows.sm};
  position: sticky;
  top: 0;
  z-index: 100;
`;

const HeaderContent = styled.div`
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 ${props => props.theme.spacing.xl};
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 64px;
`;

const Logo = styled(Link)`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  text-decoration: none;
  color: ${props => props.theme.colors.primary};
  font-weight: 700;
  font-size: 1.5rem;
  transition: color 0.2s ease;

  &:hover {
    color: ${props => props.theme.colors.primaryDark};
  }
`;

const LogoIcon = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  background: ${props => props.theme.colors.primary};
  border-radius: ${props => props.theme.borderRadius.sm};
  color: ${props => props.theme.colors.surface};
`;

const Nav = styled.nav`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.lg};

  @media (max-width: ${props => props.theme.breakpoints.mobile}) {
    display: none;
  }
`;

const NavLink = styled(Link)`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  border-radius: ${props => props.theme.borderRadius.md};
  text-decoration: none;
  color: ${props => props.theme.colors.textSecondary};
  font-weight: 500;
  transition: all 0.2s ease;
  position: relative;

  &:hover {
    color: ${props => props.theme.colors.primary};
    background: ${props => props.theme.colors.primaryLight + '10'};
  }

  ${props => props.$active && `
    color: ${props.theme.colors.primary};
    background: ${props.theme.colors.primaryLight + '15'};
    
    &::after {
      content: '';
      position: absolute;
      bottom: -1px;
      left: 50%;
      transform: translateX(-50%);
      width: 20px;
      height: 2px;
      background: ${props.theme.colors.primary};
      border-radius: 1px;
    }
  `}
`;

const UserSection = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.md};
`;

const UserInfo = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.sm};
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  background: ${props => props.theme.colors.background};
  border-radius: ${props => props.theme.borderRadius.md};
  border: 1px solid ${props => props.theme.colors.border};
`;

const UserAvatar = styled.div`
  width: 32px;
  height: 32px;
  background: ${props => props.theme.colors.primary};
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: ${props => props.theme.colors.surface};
  font-weight: 600;
  font-size: 0.875rem;
`;

const UserName = styled.span`
  font-weight: 500;
  color: ${props => props.theme.colors.text};
`;

const LogoutButton = styled.button`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
  padding: ${props => props.theme.spacing.sm} ${props => props.theme.spacing.md};
  background: none;
  border: 1px solid ${props => props.theme.colors.border};
  border-radius: ${props => props.theme.borderRadius.md};
  color: ${props => props.theme.colors.textSecondary};
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background: ${props => props.theme.colors.error + '10'};
    border-color: ${props => props.theme.colors.error};
    color: ${props => props.theme.colors.error};
  }
`;

const StatusIndicator = styled.div`
  display: flex;
  align-items: center;
  gap: ${props => props.theme.spacing.xs};
  padding: ${props => props.theme.spacing.xs} ${props => props.theme.spacing.sm};
  background: ${props => props.$status === 'online' ? props.theme.colors.success + '15' : props.theme.colors.warning + '15'};
  border: 1px solid ${props => props.$status === 'online' ? props.theme.colors.success + '30' : props.theme.colors.warning + '30'};
  border-radius: ${props => props.theme.borderRadius.sm};
  font-size: 0.75rem;
  font-weight: 500;
  color: ${props => props.$status === 'online' ? props.theme.colors.success : props.theme.colors.warning};
`;

const StatusDot = styled.div`
  width: 6px;
  height: 6px;
  background: ${props => props.$status === 'online' ? props.theme.colors.success : props.theme.colors.warning};
  border-radius: 50%;
  animation: ${props => props.$status === 'online' ? 'pulse 2s infinite' : 'none'};

  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
  }
`;

function Header() {
  const location = useLocation();
  const [systemStatus, setSystemStatus] = React.useState('online');

  // Check system status
  React.useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await fetch('/api/health');
        setSystemStatus(response.ok ? 'online' : 'offline');
      } catch (error) {
        setSystemStatus('offline');
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 30000); // Check every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('auth_token');
    window.location.href = '/';
  };

  const isActive = (path) => location.pathname === path;

  return (
    <HeaderContainer>
      <HeaderContent>
        <Logo to="/">
          <LogoIcon>
            <Zap size={20} />
          </LogoIcon>
          AgriSprayAI
        </Logo>

        <Nav>
          <NavLink to="/" $active={isActive('/')}>
            <Home size={18} />
            Home
          </NavLink>
          <NavLink to="/dashboard" $active={isActive('/dashboard')}>
            <BarChart3 size={18} />
            Dashboard
          </NavLink>
          <NavLink to="/logs" $active={isActive('/logs')}>
            <FileText size={18} />
            Logs
          </NavLink>
          <NavLink to="/settings" $active={isActive('/settings')}>
            <Settings size={18} />
            Settings
          </NavLink>
        </Nav>

        <UserSection>
          <StatusIndicator $status={systemStatus}>
            <StatusDot $status={systemStatus} />
            {systemStatus === 'online' ? 'System Online' : 'System Offline'}
          </StatusIndicator>

          <UserInfo>
            <UserAvatar>
              <Shield size={16} />
            </UserAvatar>
            <UserName>Operator</UserName>
          </UserInfo>

          <LogoutButton onClick={handleLogout}>
            <LogOut size={16} />
            Logout
          </LogoutButton>
        </UserSection>
      </HeaderContent>
    </HeaderContainer>
  );
}

export default Header;
